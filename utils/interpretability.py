# =============================================================================
# utils/interpretability.py
# Model Interpretability Analysis Module for FusionCropNetV5EDL
#
# Provides:
#   1. Grad-CAM & Grad-CAM++ — spatial explanation maps per class
#   2. Modality ablation — contribution of opt/SAR/DEM to predictions
#   3. Temporal importance — per-timestep contribution analysis
#   4. Spectral band importance — occlusion-based band sensitivity
#   5. Cross-modal attention analysis — opt↔SAR interaction strength
#   6. Feature attribution via Integrated Gradients (captum bridge)
#   7. Confusion-region analysis — where & why model confuses classes
# =============================================================================
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Grad-CAM for EDL (using alpha sum as activation signal)
# ---------------------------------------------------------------------------
class GradCAM_EDL:
    """Grad-CAM adapted for EDL heads: uses sum(alpha) or per-class alpha_k
    as the target signal, backpropagates to a chosen intermediate layer."""

    def __init__(self, model, target_layer_name="decoder"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self._hook_handles = []

    def _save_activation(self, module, inp, outp):
        self.activations = outp

    def _save_gradient(self, module, inp, outp):
        self.gradients = outp[0]

    def _resolve_layer(self):
        """Resolve a named submodule, e.g. 'decoder.merge2' or 'decoder'."""
        parts = self.target_layer_name.split(".")
        module = self.model
        for p in parts:
            module = getattr(module, p, None)
            if module is None:
                raise ValueError(f"Cannot resolve layer: {self.target_layer_name}")
        return module

    def _register_hooks(self):
        layer = self._resolve_layer()
        self._hook_handles.append(
            layer.register_forward_hook(self._save_activation))
        self._hook_handles.append(
            layer.register_full_backward_hook(self._save_gradient))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def __call__(self, opt_seq, sar_seq, dem, doy, class_idx=None):
        """Compute Grad-CAM heatmap.

        Args:
            opt_seq, sar_seq, dem, doy: model inputs (B, ...)
            class_idx: target class for CAM. If None, uses argmax(alpha).

        Returns:
            heatmap: (B, H, W) numpy array in [0, 1]
        """
        self._register_hooks()
        try:
            alpha = self.model(opt_seq, sar_seq, dem, doy)
            if isinstance(alpha, tuple):
                alpha = alpha[0]

            if class_idx is None:
                class_idx = alpha.sum(dim=1).argmax(dim=1)  # per-sample

            if isinstance(class_idx, torch.Tensor) and class_idx.ndim == 1:
                target = alpha[range(len(class_idx)), class_idx].sum()
            else:
                target = alpha[:, class_idx].sum()

            self.model.zero_grad()
            target.backward()

            acts = self.activations.detach()  # (B, C, h, w)
            grads = self.gradients.detach()   # (B, C, h, w)

            weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            cam = (weights * acts).sum(dim=1, keepdim=True)  # (B, 1, h, w)
            cam = F.relu(cam)

            # Resize to input spatial size
            H_in, W_in = opt_seq.shape[-2], opt_seq.shape[-1]
            cam = F.interpolate(cam, (H_in, W_in), mode='bilinear',
                                align_corners=False)
            heatmaps = []
            for b in range(cam.shape[0]):
                h = cam[b, 0]
                h = (h - h.min()) / (h.max() - h.min() + 1e-8)
                heatmaps.append(h.cpu().numpy())
            return np.stack(heatmaps) if len(heatmaps) > 1 else heatmaps[0]
        finally:
            self.remove_hooks()


def gradcam_per_class(model, opt_seq, sar_seq, dem, doy, num_classes=7,
                      target_layer="decoder"):
    """Compute Grad-CAM for each class in a single pass loop."""
    gradcam = GradCAM_EDL(model, target_layer_name=target_layer)
    maps = {}
    for k in range(num_classes):
        try:
            hm = gradcam(opt_seq, sar_seq, dem, doy, class_idx=k)
            maps[k] = hm
        except Exception:
            maps[k] = np.zeros((opt_seq.shape[-2], opt_seq.shape[-1]),
                               dtype=np.float32)
    return maps


# ---------------------------------------------------------------------------
# Modality ablation analysis
# ---------------------------------------------------------------------------
def modality_ablation(model, opt_seq, sar_seq, dem, doy,
                      num_classes=7, device="cpu"):
    """Measure prediction change when removing each modality.

    Returns per-modality contribution as the OA drop or probability shift.
    """
    from models.fusion_net_v5_edl import dirichlet_to_predictions

    model.eval()
    B = opt_seq.shape[0]

    # Resolve device from model
    if isinstance(device, str):
        device = next(model.parameters()).device

    input_args = (opt_seq, sar_seq, dem, doy)

    with torch.no_grad():
        alpha_full = model(*input_args)
        if isinstance(alpha_full, tuple):
            alpha_full = alpha_full[0]
        preds_full = dirichlet_to_predictions(alpha_full)
        probs_full = preds_full["probs"]
        pred_full = preds_full["pred_class"]
        vacuity_full = preds_full["vacuity"]

    # Placeholder tensors
    z_opt = torch.zeros_like(opt_seq)
    z_sar = torch.zeros_like(sar_seq)
    z_dem = torch.zeros_like(dem)

    configs = {
        "full": (True, True, True),
        "no_opt": (False, True, True),
        "no_sar": (True, False, True),
        "no_dem": (True, True, False),
    }

    results = {}
    for name, (use_opt, use_sar, use_dem) in configs.items():
        with torch.no_grad():
            if name == "full":
                alpha_abl = alpha_full
            else:
                pre_head = model._encode(
                    opt_seq if use_opt else z_opt,
                    sar_seq if use_sar else z_sar,
                    dem if use_dem else z_dem,
                    doy, None, None,
                    modality_mask=(use_opt, use_sar, use_dem),
                )[0]
                alpha_abl = model.edl_head(pre_head)
            preds_abl = dirichlet_to_predictions(alpha_abl)
            pred_abl = preds_abl["pred_class"]
            probs_abl = preds_abl["probs"]

        # Agreement with full model
        agree = (pred_abl == pred_full).float().mean()
        # Probability shift (Jensen-Shannon approximation)
        kl = (probs_full * (torch.log(probs_full + 1e-8)
                            - torch.log(probs_abl + 1e-8))).sum(dim=1).mean()
        # Mean vacuity change
        vac_shift = (preds_abl["vacuity"] - vacuity_full).abs().mean()

        results[name] = {
            "agreement": float(agree.cpu()),
            "prob_shift_kl": float(kl.cpu()),
            "vacuity_shift": float(vac_shift.cpu()),
        }

    # Compute relative contributions
    opt_contrib = 1.0 - results["no_opt"]["agreement"]
    sar_contrib = 1.0 - results["no_sar"]["agreement"]
    dem_contrib = 1.0 - results["no_dem"]["agreement"]
    total = opt_contrib + sar_contrib + dem_contrib + 1e-10

    results["relative_importance"] = {
        "optical": float(opt_contrib / total),
        "sar": float(sar_contrib / total),
        "dem": float(dem_contrib / total),
    }

    return results


# ---------------------------------------------------------------------------
# Temporal importance analysis
# ---------------------------------------------------------------------------
def temporal_importance(model, opt_seq, sar_seq, dem, doy, device="cpu"):
    """Analyze per-timestep importance by zeroing each step and measuring
    the resulting alpha shift.

    Returns:
        importance: (T,) array of importance scores
        step_details: list of per-step metric dicts
    """
    model.eval()
    T = opt_seq.shape[1]

    with torch.no_grad():
        alpha_full = model(opt_seq, sar_seq, dem, doy)
        if isinstance(alpha_full, tuple):
            alpha_full = alpha_full[0]
        pred_full = alpha_full.argmax(dim=1)  # (B, H, W)

    importance = np.zeros(T)
    details = []

    for t in range(T):
        opt_masked = opt_seq.clone()
        sar_masked = sar_seq.clone()
        opt_masked[:, t] = 0.0
        sar_masked[:, t] = 0.0

        with torch.no_grad():
            alpha_mask = model(opt_masked, sar_masked, dem, doy)
            if isinstance(alpha_mask, tuple):
                alpha_mask = alpha_mask[0]
            pred_mask = alpha_mask.argmax(dim=1)

        delta = (pred_full != pred_mask).float().mean().item()
        importance[t] = delta
        details.append({"step": t, "prediction_flip_rate": float(delta)})

    importance = importance / (importance.sum() + 1e-10)
    return importance, details


def temporal_entropy_analysis(model, opt_seq, sar_seq, dem, doy, device="cpu"):
    """Compute how each timestep affects prediction entropy.

    Lower entropy after adding a step = that step resolves ambiguity.
    """
    from models.fusion_net_v5_edl import dirichlet_to_predictions

    model.eval()
    T = opt_seq.shape[1]
    entropies = np.zeros(T)

    with torch.no_grad():
        for t in range(T):
            opt_step = opt_seq[:, t:t+1]
            sar_step = sar_seq[:, t:t+1]
            doy_step = doy[:, t:t+1]

            alpha = model(opt_step, sar_step, dem, doy_step)
            if isinstance(alpha, tuple):
                alpha = alpha[0]
            preds = dirichlet_to_predictions(alpha)
            probs = preds["probs"]
            ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
            entropies[t] = ent

    return entropies


# ---------------------------------------------------------------------------
# Spectral band importance via occlusion
# ---------------------------------------------------------------------------
def spectral_band_importance(model, opt_seq, sar_seq, dem, doy,
                             device="cpu", n_samples=100):
    """Occlusion-based band importance for optical channels.

    Zeros one band at a time, measures prediction change probability.
    """
    model.eval()
    C_opt = opt_seq.shape[2]
    C_sar = sar_seq.shape[2]

    with torch.no_grad():
        alpha_full = model(opt_seq, sar_seq, dem, doy)
        if isinstance(alpha_full, tuple):
            alpha_full = alpha_full[0]
        pred_full = alpha_full.argmax(dim=1)

    imp_opt = np.zeros(C_opt)
    imp_sar = np.zeros(C_sar)

    for c in range(C_opt):
        opt_occ = opt_seq.clone()
        opt_occ[:, :, c] = 0.0
        with torch.no_grad():
            alpha_occ = model(opt_occ, sar_seq, dem, doy)
            if isinstance(alpha_occ, tuple):
                alpha_occ = alpha_occ[0]
            pred_occ = alpha_occ.argmax(dim=1)
        imp_opt[c] = (pred_full != pred_occ).float().mean().item()

    for c in range(C_sar):
        sar_occ = sar_seq.clone()
        sar_occ[:, :, c] = 0.0
        with torch.no_grad():
            alpha_occ = model(opt_seq, sar_occ, dem, doy)
            if isinstance(alpha_occ, tuple):
                alpha_occ = alpha_occ[0]
            pred_occ = alpha_occ.argmax(dim=1)
        imp_sar[c] = (pred_full != pred_occ).float().mean().item()

    imp_opt = imp_opt / (imp_opt.sum() + 1e-10)
    imp_sar = imp_sar / (imp_sar.sum() + 1e-10)

    return {
        "optical_bands": imp_opt.tolist(),
        "sar_bands": imp_sar.tolist(),
    }


# ---------------------------------------------------------------------------
# Cross-modal attention analysis
# ---------------------------------------------------------------------------
def cross_modal_attention_analysis(model, opt_seq, sar_seq, dem, doy, device="cpu"):
    """Analyze cross-modal attention flow strength between optical and SAR.

    Hooks into CrossModalAttention to capture gate values and attention maps.
    Returns per-pixel average gate value (opt→SAR weight) as a heatmap.
    """
    gate_values = []
    attn_outputs = []

    def gate_hook(module, inp, outp):
        B, C, H, W = outp.shape
        if H > 1 and W > 1:
            gate_values.append(outp.mean(dim=1).detach().cpu())  # (B, H, W)

    def attn_hook(module, inp, outp):
        attn_outputs.append(outp.detach().cpu())

    hooks = []
    for name, mod in model.named_modules():
        if "cross_modal.gate" in name and isinstance(mod, torch.nn.Sequential):
            hooks.append(mod.register_forward_hook(gate_hook))
        if "cross_modal.proj" in name:
            hooks.append(mod.register_forward_hook(attn_hook))

    model.eval()
    with torch.no_grad():
        model(opt_seq, sar_seq, dem, doy)

    for h in hooks:
        h.remove()

    result = {}
    if gate_values:
        gate_stack = torch.stack(gate_values, dim=0).mean(dim=0)
        result["gate_mean"] = gate_stack[0].numpy()  # (H, W)
    if attn_outputs:
        result["cross_modal_output_norm"] = attn_outputs[-1].norm(
            dim=1).mean(dim=0).numpy()

    return result


# ---------------------------------------------------------------------------
# Confusion-region analysis
# ---------------------------------------------------------------------------
def confusion_region_analysis(alpha, targets, num_classes=7, ignore_index=255):
    """Identify spatial regions where the model is most confused.

    For each pair of classes (i, j), find pixels where:
      - True class = i, but second-highest probability = j (or vice versa)
      - Report the mean vacuity, dissonance in those regions.

    Returns a per-class-pair confusion matrix with uncertainty info.
    """
    if alpha.ndim == 4:
        B, K, H, W = alpha.shape
        alpha = alpha.transpose(0, 2, 3, 1).reshape(-1, K)
    if targets.ndim == 3:
        targets = targets.reshape(-1)
    elif targets.ndim == 2:
        targets = targets.reshape(-1)

    valid = targets != ignore_index
    alpha_v = alpha[valid]
    tgt_v = targets[valid]

    K = num_classes
    S = alpha_v.sum(axis=-1, keepdims=True)
    probs = alpha_v / S
    vacuity = K / S.squeeze(-1)
    dissonance = 1.0 - (probs * probs).sum(axis=-1)

    preds = probs.argmax(axis=-1)
    top2 = np.argsort(-probs, axis=-1)[:, :2]

    confusion_pairs = {}
    for i in range(K):
        for j in range(K):
            if i >= j:
                continue

            # Pixels where true=i and top2 contains j as runner-up
            mask_ij = (tgt_v == i) & (top2[:, 1] == j)
            mask_ji = (tgt_v == j) & (top2[:, 1] == i)
            mask = mask_ij | mask_ji
            n_confused = mask.sum()

            if n_confused < 5:
                confusion_pairs[f"{i}_{j}"] = {
                    "n": int(n_confused),
                    "mean_vacuity": None,
                    "mean_dissonance": None,
                }
                continue

            confusion_pairs[f"{i}_{j}"] = {
                "n": int(n_confused),
                "mean_vacuity": float(vacuity[mask].mean()),
                "mean_dissonance": float(dissonance[mask].mean()),
                "frac": float(n_confused / len(tgt_v)),
                "accuracy_on_pair": float((preds[mask] == tgt_v[mask]).mean()),
            }

    return confusion_pairs


# ---------------------------------------------------------------------------
# Pixel-level explanation report
# ---------------------------------------------------------------------------
def pixel_explanation_report(alpha, targets, num_classes=7, ignore_index=255):
    """For each correctly vs incorrectly classified pixel, report the
    distribution of vacuity, dissonance, top-2 probability margin, and
    entropy.

    Useful for diagnosing: "Do errors correspond to high uncertainty?"
    """
    if alpha.ndim == 4:
        B, K, H, W = alpha.shape
        alpha = alpha.transpose(0, 2, 3, 1).reshape(-1, K)
    if targets.ndim == 3:
        targets = targets.reshape(-1)
    elif targets.ndim == 2:
        targets = targets.reshape(-1)

    valid = targets != ignore_index
    alpha_v = alpha[valid]
    tgt_v = targets[valid]

    K = num_classes
    S = alpha_v.sum(axis=-1, keepdims=True)
    probs = alpha_v / S
    vacuity = K / S.squeeze(-1)
    dissonance = 1.0 - (probs * probs).sum(axis=-1)

    preds = probs.argmax(axis=-1)
    correct = preds == tgt_v

    sorted_probs = np.sort(probs, axis=-1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=-1)
    conf = sorted_probs[:, 0]

    def _stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    return {
        "correct": {
            "n": int(correct.sum()),
            "vacuity": _stats(vacuity[correct]),
            "dissonance": _stats(dissonance[correct]),
            "margin": _stats(margin[correct]),
            "entropy": _stats(entropy[correct]),
            "confidence": _stats(conf[correct]),
        },
        "incorrect": {
            "n": int((~correct).sum()),
            "vacuity": _stats(vacuity[~correct]),
            "dissonance": _stats(dissonance[~correct]),
            "margin": _stats(margin[~correct]),
            "entropy": _stats(entropy[~correct]),
            "confidence": _stats(conf[~correct]),
        },
    }


# ---------------------------------------------------------------------------
# Captum Integrated Gradients bridge (optional, requires captum)
# ---------------------------------------------------------------------------
def integrated_gradients_attribution(model, opt_seq, sar_seq, dem, doy,
                                     class_idx=None, n_steps=20,
                                     internal_batch_size=1, device="cpu"):
    """Integrated Gradients feature attribution using Captum.

    Requires: pip install captum

    Attributes prediction to each input pixel, returning per-modality
    attribution maps.

    Args:
        model: FusionCropNetV5EDL
        opt_seq, sar_seq, dem, doy: input tensors
        class_idx: target class; defaults to argmax
        n_steps: IG steps

    Returns:
        dict with 'opt_attr', 'sar_attr', 'dem_attr' attribution tensors
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise ImportError("captum required: pip install captum")

    model.eval()

    def _forward_wrapper(*inputs):
        # inputs order: opt, sar, dem, doy
        alpha = model(inputs[0], inputs[1], inputs[2], inputs[3])
        if isinstance(alpha, tuple):
            alpha = alpha[0]
        # Use mean of alpha over spatial dims as scalar output
        return alpha.mean(dim=(2, 3)).sum(dim=1)  # (B,)

    ig = IntegratedGradients(_forward_wrapper)

    baselines = (
        torch.zeros_like(opt_seq),
        torch.zeros_like(sar_seq),
        torch.zeros_like(dem),
        doy,  # keep doy fixed
    )

    attributions = ig.attribute(
        (opt_seq, sar_seq, dem, doy),
        baselines=baselines,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )

    return {
        "opt_attr": attributions[0].detach(),
        "sar_attr": attributions[1].detach(),
        "dem_attr": attributions[2].detach(),
    }
