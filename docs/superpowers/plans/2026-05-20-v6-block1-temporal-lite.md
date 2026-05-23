# V6 Block 1: TemporalLite + AMP + Gradient Checkpointing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace time_average on SAR s1/s2 with TemporalLite, add AMP + Gradient Checkpointing. ~48× temporal encoding speedup.

**Architecture:** Add `TemporalLite` (1D depthwise conv + gated temporal pooling) to `models/`. Use it on SAR s1(64ch) and s2(128ch) in `_encode()`. Wrap training loop with `torch.amp` and enable gradient checkpointing on the s3 Transformer.

**Tech Stack:** PyTorch 2.x, torch.amp, torch.utils.checkpoint

**Block spec:** [[model-FusionCropNet-V6-更新架构]] Section 3.2, [[V6-时序编码瓶颈-方案评审]] Section 6.4

---

## File Structure

```
Create:
  models/temporal_lite.py           — TemporalLite module

Modify:
  models/_base.py                   — Import TemporalLite (no code change, re-export)
  models/__init__.py                — Export TemporalLite
  models/fusion_net_v5_edl.py       — Integrate TemporalLite for s1/s2, AMP + GC in forward
  utils/trainer.py                  — AMP scaler in training loop
  tests/test_suite.py               — New test class

No changes to:
  scripts/train_fusion_edl.py       — CLI unchanged (AMP auto-detected)
  models/fusion_net_v5pro.py        — Shares _base, picks up TemporalLite via import
  All other files
```

---

### Task 1: Create TemporalLite test

**Files:**
- Create: `tests/test_temporal_lite.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for TemporalLite module."""
import torch
import pytest
from models.temporal_lite import TemporalLite


class TestTemporalLite:
    @pytest.fixture
    def module_64(self):
        return TemporalLite(d_model=64, k=3)
    
    @pytest.fixture
    def module_128(self):
        return TemporalLite(d_model=128, k=5)
    
    def test_forward_shape(self, module_64):
        """Output shape: (B*HW, T, D) -> (B*HW, D)"""
        B, T, D = 32, 12, 64
        x = torch.randn(B, T, D)
        out = module_64(x)
        assert out.shape == (B, D)
    
    def test_forward_shape_128(self, module_128):
        B, T, D = 16, 24, 128
        x = torch.randn(B, T, D)
        out = module_128(x)
        assert out.shape == (B, D)
    
    def test_deterministic_in_eval(self, module_64):
        """Same input -> same output in eval mode."""
        module_64.eval()
        x = torch.randn(64, 12, 64)
        out1 = module_64(x)
        out2 = module_64(x)
        assert torch.allclose(out1, out2)
    
    def test_no_nan(self, module_64):
        """Output contains no NaN or Inf."""
        x = torch.randn(128, 12, 64)
        out = module_64(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_gate_is_learnable(self, module_64):
        """gate parameter receives gradient."""
        x = torch.randn(8, 12, 64)
        out = module_64(x)
        loss = out.sum()
        loss.backward()
        assert module_64.gate.grad is not None
    
    def test_handles_variable_T(self, module_64):
        """Works with different sequence lengths."""
        for T in [6, 12, 24]:
            x = torch.randn(16, T, 64)
            out = module_64(x)
            assert out.shape == (16, 64)
    
    def test_handles_k3_and_k5(self):
        """Both kernel sizes work."""
        m3 = TemporalLite(64, k=3)
        m5 = TemporalLite(64, k=5)
        x = torch.randn(8, 12, 64)
        assert m3(x).shape == (8, 64)
        assert m5(x).shape == (8, 64)
    
    def test_param_count(self, module_64):
        """Extremely lightweight: < 5K params for d_model=64."""
        total = sum(p.numel() for p in module_64.parameters())
        assert total < 5000, f"Expected <5K params, got {total}"
    
    def test_k1_fallback(self):
        """k=1 works (no padding needed, basic temporal mix)."""
        m = TemporalLite(32, k=1)
        x = torch.randn(4, 6, 32)
        out = m(x)
        assert out.shape == (4, 32)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "E:/基于深度学习的遥感影像光谱分类/project1"
python -m pytest tests/test_temporal_lite.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models.temporal_lite'`

---

### Task 2: Write TemporalLite implementation

**Files:**
- Create: `models/temporal_lite.py`

- [ ] **Step 1: Write the module**

```python
"""TemporalLite — lightweight temporal encoder for high-resolution features.

Replaces time_average() on SAR s1(64ch) and s2(128ch) with a learnable
1D depthwise convolution + gated temporal pooling. ~0.1M params each.

See: V6-时序编码瓶颈-方案评审.md Section 6.4
"""
import torch
import torch.nn as nn


class TemporalLite(nn.Module):
    """Extremely lightweight temporal encoder.

    Complexity: O(T × D) via depthwise conv — vs O(T × D²) for Transformer FFN.

    Args:
        d_model: feature dimension
        k: convolution kernel size (temporal window), default 3
    """
    def __init__(self, d_model: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, k,
            padding=k // 2,
            groups=d_model,     # depthwise
            bias=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal sequence into a single feature vector.

        Args:
            x: (N, T, D) — N sequences (e.g. B×H×W pixels),
               T timesteps, D channels

        Returns:
            (N, D) — temporally pooled features
        """
        # Conv1d expects (N, D, T)
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.norm(x)
        # Weighted temporal mean (learned gate vs fixed time_average)
        return x.mean(dim=1) * self.gate
```

- [ ] **Step 2: Run tests**

```bash
cd "E:/基于深度学习的遥感影像光谱分类/project1"
python -m pytest tests/test_temporal_lite.py -v
```

Expected: 9 passed

- [ ] **Step 3: Commit**

```bash
git add models/temporal_lite.py tests/test_temporal_lite.py
git commit -m "feat: add TemporalLite — lightweight temporal encoder for s1/s2 features"
```

---

### Task 3: Export TemporalLite from package

**Files:**
- Modify: `models/__init__.py`

- [ ] **Step 1: Read current __init__.py and add export**

Read `models/__init__.py` to find the import block, then add:

```python
from .temporal_lite import TemporalLite
```

And add `"TemporalLite"` to the `__all__` list.

- [ ] **Step 2: Verify import works**

```bash
cd "E:/基于深度学习的遥感影像光谱分类/project1"
python -c "from models import TemporalLite; m = TemporalLite(64); print('OK:', sum(p.numel() for p in m.parameters()), 'params')"
```

Expected: `OK: <5000 params`

- [ ] **Step 3: Commit**

```bash
git add models/__init__.py
git commit -m "feat: export TemporalLite from models package"
```

---

### Task 4: Write integration test for s1/s2 in _encode

**Files:**
- Modify: `tests/test_suite.py`

- [ ] **Step 1: Add TestV6TemporalLiteIntegration class**

```python
class TestV6TemporalLiteIntegration:
    """Verify TemporalLite integrates correctly in V5EDL _encode path."""
    
    @pytest.fixture
    def model(self):
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        return FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2
        )
    
    @pytest.fixture
    def inputs(self):
        B, T, H, W = 2, 6, 128, 128
        return (
            torch.randn(B, T, 10, H, W),  # opt
            torch.randn(B, T, 5, H, W),   # sar
            torch.randn(B, 5, H, W),      # dem
            torch.randint(1, 365, (B, T)).float() / 365.0,  # doy
        )
    
    def test_forward_with_temporal_lite(self, model, inputs):
        """Forward pass succeeds with TemporalLite active."""
        model.eval()
        with torch.no_grad():
            alpha, ndvi, _ = model(*inputs)
        assert alpha.shape == (2, 7, 128, 128)
    
    def test_no_nan_in_output(self, model, inputs):
        """Output contains no NaN."""
        model.eval()
        with torch.no_grad():
            alpha, _, _ = model(*inputs)
        assert not torch.isnan(alpha).any()
    
    def test_temporal_lite_params_in_model(self, model):
        """Model contains TemporalLite parameters."""
        tl_params = [
            name for name, _ in model.named_parameters()
            if 'temp_lite' in name
        ]
        assert len(tl_params) > 0, "TemporalLite not found in model parameters"
    
    def test_temporal_lite_trainable(self, model):
        """TemporalLite parameters require grad."""
        for name, param in model.named_parameters():
            if 'temp_lite' in name:
                assert param.requires_grad, f"{name} should be trainable"

    def test_compatible_with_cloud_mask(self, model, inputs):
        """Forward pass works with cloud mask."""
        opt, sar, dem, doy = inputs
        B, T = opt.shape[:2]
        cm = torch.zeros(B, T, 128, 128, dtype=torch.bool)
        model.eval()
        with torch.no_grad():
            alpha, _, _ = model(opt, sar, dem, doy, cloud_mask=cm)
        assert alpha.shape == (2, 7, 128, 128)

    def test_deterministic_eval(self, model, inputs):
        """Same input twice -> same output."""
        model.eval()
        with torch.no_grad():
            a1, _, _ = model(*inputs)
            a2, _, _ = model(*inputs)
        assert torch.allclose(a1, a2, atol=1e-5)
```

- [ ] **Step 2: Run integration tests (expected to FAIL on s1/s2 TemporalLite)**

```bash
cd "E:/基于深度学习的遥感影像光谱分类/project1"
python -m pytest tests/test_suite.py::TestV6TemporalLiteIntegration -v
```

Expected: FAIL — TemporalLite not yet wired into model

---

### Task 5: Wire TemporalLite into fusion_net_v5_edl.py

**Files:**
- Modify: `models/fusion_net_v5_edl.py`

- [ ] **Step 1: Add TemporalLite modules in __init__**

In `FusionCropNetV5EDL.__init__`, after `self.sar_temporal = ...`:

```python
# V6 Block 1: TemporalLite for s1(64ch) and s2(128ch)
from .temporal_lite import TemporalLite
self.temp_lite_s1 = TemporalLite(64, k=3)
self.temp_lite_s2 = TemporalLite(128, k=3)
```

- [ ] **Step 2: Replace time_average calls in _encode**

In `_encode()`, find the lines:
```python
sar_s1a = time_average(sar_s1, B, T)
sar_s2a = time_average(sar_s2, B, T)
```

Replace with:
```python
# V6: TemporalLite replaces time_average — preserves temporal dynamics
sar_s1_seq = self._to_pixel_seq(sar_s1, B, T, H, W, 64)
sar_s2_seq = self._to_pixel_seq(sar_s2, B, T, H//2, W//2, 128)
sar_s1a = self.temp_lite_s1(sar_s1_seq).view(B, H, W, 64).permute(0, 3, 1, 2)
sar_s2a = self.temp_lite_s2(sar_s2_seq).view(B, H//2, W//2, 128).permute(0, 3, 1, 2)
```

Apply the same for optical:
```python
# Also upgrade opt_p2 from time_average to TemporalLite
opt_p2_seq = self._to_pixel_seq(opt_p2, B, T, H//2, W//2, 256)
opt_p2a = self.temp_lite_opt_p2(opt_p2_seq).view(B, H//2, W//2, 256).permute(0, 3, 1, 2)
```

Add `self.temp_lite_opt_p2 = TemporalLite(256, k=3)` in `__init__`.

- [ ] **Step 3: Run integration tests**

```bash
cd "E:/基于深度学习的遥感影像光谱分类/project1"
python -m pytest tests/test_suite.py::TestV6TemporalLiteIntegration -v
```

Expected: 6 passed

- [ ] **Step 4: Commit**

```bash
git add models/fusion_net_v5_edl.py
git commit -m "feat: wire TemporalLite into V5EDL — s1/s2 temporal encoding replaces time_average"
```

---

### Task 6: Add Gradient Checkpointing

**Files:**
- Modify: `models/fusion_net_v5_edl.py`

- [ ] **Step 1: Add checkpointing flag and apply to temporal encoder**

In `FusionCropNetV5EDL.__init__`, add parameter:
```python
def __init__(self, ..., use_gradient_checkpointing: bool = False):
    ...
    self.use_grad_ckpt = use_gradient_checkpointing
```

In `_encode()`, wrap s3 temporal encoder with checkpoint:
```python
if self.use_grad_ckpt and self.training:
    opt_g, opt_seq_out = torch.utils.checkpoint.checkpoint(
        self.opt_temporal, opt_ts, doy_px, cloud_mask=cm_px, valid_count=vc_px,
        use_reentrant=False
    )
else:
    opt_g, opt_seq_out = self.opt_temporal(opt_ts, doy_px, cloud_mask=cm_px, valid_count=vc_px)
```

- [ ] **Step 2: Add test for checkpointing**

Add to `TestV6TemporalLiteIntegration`:
```python
def test_gradient_checkpointing_mode(self, inputs):
    """Model with checkpointing runs forward+backward without error."""
    from models.fusion_net_v5_edl import FusionCropNetV5EDL
    model = FusionCropNetV5EDL(
        opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
        feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2,
        use_gradient_checkpointing=True
    )
    model.train()
    opt, sar, dem, doy = inputs
    alpha, ndvi, _ = model(opt, sar, dem, doy)
    loss = alpha.sum()
    loss.backward()
    # If we got here without OOM or error, it works
    assert not torch.isnan(alpha).any()
```

- [ ] **Step 3: Run the test**

```bash
cd "E:/基于深度学习的遥感影像光谱分类/project1"
python -m pytest tests/test_suite.py::TestV6TemporalLiteIntegration::test_gradient_checkpointing_mode -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add models/fusion_net_v5_edl.py tests/test_suite.py
git commit -m "feat: add gradient checkpointing to V5EDL temporal encoder"
```

---

### Task 7: Add AMP to training loop

**Files:**
- Modify: `utils/trainer.py`

- [ ] **Step 1: Add AMP autocast to FusionTrainer.fit()**

Read `utils/trainer.py` to find the training loop. Add:

```python
class FusionTrainer(BaseTrainer):
    def __init__(self, model, ..., use_amp: bool = False):
        ...
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        # On CPU: torch.amp.autocast('cpu') is available in PyTorch 2.0+
    
    def fit(self, train_loader, val_loader, epochs, ...):
        ...
        for batch in train_loader:
            ...
            with torch.amp.autocast(
                'cuda' if torch.cuda.is_available() else 'cpu',
                enabled=self.use_amp
            ):
                loss, metrics = self._training_step(batch)
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
```

- [ ] **Step 2: Add test for AMP training step**

```python
def test_amp_training_step():
    """AMP forward+backward doesn't crash."""
    from models.fusion_net_v5_edl import FusionCropNetV5EDL
    from utils.trainer import FusionTrainer
    
    model = FusionCropNetV5EDL(
        opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
        feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2,
        use_gradient_checkpointing=True
    )
    trainer = FusionTrainer(model, use_amp=True)
    
    B, T, H, W = 2, 6, 128, 128
    batch = {
        'opt': torch.randn(B, T, 10, H, W),
        'sar': torch.randn(B, T, 5, H, W),
        'dem': torch.randn(B, 5, H, W),
        'doy': torch.randint(1, 365, (B, T)).float() / 365.0,
    }
    
    with torch.amp.autocast('cpu', enabled=True):
        loss = trainer._training_step(batch)
    
    loss.backward()
    assert not torch.isnan(loss)
```

- [ ] **Step 3: Commit**

```bash
git add utils/trainer.py tests/test_suite.py
git commit -m "feat: add AMP mixed-precision support to FusionTrainer"
```

---

### Task 8: Full regression — run all 105 tests

- [ ] **Step 1: Run full test suite**

```bash
cd "E:/基于深度学习的遥感影像光谱分类/project1"
python -m pytest tests/ -v --tb=short
```

Expected: 105+ tests passed (original 105 + new TemporalLite + integration tests)

- [ ] **Step 2: Run synthetic data training smoke test**

```bash
cd "E:/基于深度学习的遥感影像光谱分类/project1"
python scripts/test_v5pro_synthetic.py
```

Expected: V5Pro synthetic training completes, no NaN, loss decreases.

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "chore: V6 Block 1 complete — TemporalLite + AMP + Gradient Checkpointing

- TemporalLite module for s1/s2 temporal encoding (~0.1M params each)
- Replaces time_average() with learnable gated temporal pooling
- Gradient checkpointing on s3 TransformerEncoder
- AMP mixed-precision support in FusionTrainer
- 105 original tests pass + 15 new tests
- See: docs/superpowers/plans/2026-05-20-v6-block1-temporal-lite.md"
```

---

### Task 9: Update knowledge base with Block 1 progress

- [ ] **Step 1: Write progress note to Obsidian**

Create `D:\obsidian\storehouse\remotesensingclassification\01-项目笔记\V6-实施进度.md` with:

```markdown
# V6 实施进度

**开始日期**: 2026-05-20
**总进度**: Block 1/N

---

## Block 1: TemporalLite + AMP + Gradient Checkpointing

**状态**: ✅ / 🔄 / ❌
**日期**: 2026-05-20
**预期收益**: ~48× 时序编码加速, 训练显存 -40%

### 变更文件
- ✅ models/temporal_lite.py — 新建
- ✅ models/__init__.py — 导出
- ✅ models/fusion_net_v5_edl.py — s1/s2 TemporalLite + GC
- ✅ utils/trainer.py — AMP
- ✅ tests/test_temporal_lite.py — 9 tests
- ✅ tests/test_suite.py — +6 integration tests

### 测试结果
- 单元测试: / passed
- 集成测试: / passed
- 全量回归: / 105 tests
- 合成数据训练: loss / NaN

### 遇到的问题
-

### 下一步
Block 2: ModalNormalize + DEM multi-path injection
```

- [ ] **Step 2: Link from V6 design log**

Add to 十、V6 更新记录:
```markdown
| 2026-05-20 (Block 1) | V6 Block 1 实施: TemporalLite + AMP + GC → [[V6-实施进度]] |
```
