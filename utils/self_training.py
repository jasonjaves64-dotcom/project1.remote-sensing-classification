"""Self-training with EDL vacuity-based pseudo-label filtering.

Uses model uncertainty (vacuity) to select high-confidence predictions
on unlabeled data, adding them to the training set iteratively.
"""
import torch
import torch.nn.functional as F


def filter_by_vacuity(alpha: torch.Tensor, vacuity_threshold: float = 0.3,
                       confidence_threshold: float = 0.9) -> torch.Tensor:
    """Select pixels with low vacuity (high evidence) AND high confidence.

    Args:
        alpha: (B, K, H, W) Dirichlet parameters from EDL head
        vacuity_threshold: max vacuity to accept
        confidence_threshold: min confidence to accept

    Returns:
        mask: (B, H, W) bool — True = selected as pseudo-label
    """
    K = alpha.shape[1]
    S = alpha.sum(dim=1)  # (B, H, W)
    vacuity = K / S  # (B, H, W)
    probs = alpha / S.unsqueeze(1)  # (B, K, H, W)
    confidence, pred = probs.max(dim=1)  # (B, H, W)

    low_vacuity = vacuity < vacuity_threshold
    high_conf = confidence > confidence_threshold

    return low_vacuity & high_conf


def generate_pseudo_labels(model, unlabeled_loader, device='cpu',
                            vacuity_threshold=0.3, confidence_threshold=0.9):
    """Generate pseudo-labels for unlabeled data.

    Args:
        model: FusionCropNetV5EDL in eval mode
        unlabeled_loader: DataLoader yielding batches with 'opt','sar','dem','doy'
        device: torch device
        vacuity_threshold: max vacuity for acceptance
        confidence_threshold: min confidence for acceptance

    Returns:
        list of dicts: each with 'opt','sar','dem','doy','pseudo_label','mask'
    """
    model.eval()
    pseudo_samples = []

    with torch.no_grad():
        for batch in unlabeled_loader:
            opt = batch['opt'].to(device)
            sar = batch['sar'].to(device)
            dem = batch['dem'].to(device)
            doy = batch['doy'].to(device)

            alpha, _, _ = model(opt, sar, dem, doy)
            mask = filter_by_vacuity(alpha, vacuity_threshold, confidence_threshold)

            if mask.any():
                probs = alpha / alpha.sum(dim=1, keepdim=True)
                pseudo_label = probs.argmax(dim=1)
                pseudo_samples.append({
                    'opt': opt.cpu(),
                    'sar': sar.cpu(),
                    'dem': dem.cpu(),
                    'doy': doy.cpu(),
                    'pseudo_label': pseudo_label.cpu(),
                    'mask': mask.cpu(),
                })

    return pseudo_samples


class SelfTrainingLoop:
    """Iterative self-training with vacuity-based filtering.

    Each round:
      1. Generate pseudo-labels on unlabeled data
      2. Filter by vacuity + confidence
      3. Add to training set
      4. Retrain model

    Args:
        model: the model to self-train
        vacuity_threshold: initial vacuity threshold (relaxed over rounds)
        confidence_threshold: initial confidence threshold (relaxed over rounds)
        max_rounds: maximum self-training rounds
    """
    def __init__(self, model, vacuity_threshold=0.3, confidence_threshold=0.9,
                 max_rounds=3):
        self.model = model
        self.vacuity_threshold = vacuity_threshold
        self.confidence_threshold = confidence_threshold
        self.max_rounds = max_rounds
        self.round_history = []  # list of (n_added, vacuity_threshold_used)

    def run_round(self, unlabeled_loader, device='cpu'):
        """Execute one self-training round. Returns pseudo-labeled samples."""
        pseudo = generate_pseudo_labels(
            self.model, unlabeled_loader, device,
            self.vacuity_threshold, self.confidence_threshold
        )
        n_added = sum(p['mask'].sum().item() for p in pseudo)
        self.round_history.append((n_added, self.vacuity_threshold))

        # Relax thresholds for next round (accept more samples over time)
        self.vacuity_threshold = min(0.5, self.vacuity_threshold + 0.05)
        self.confidence_threshold = max(0.7, self.confidence_threshold - 0.05)

        return pseudo
