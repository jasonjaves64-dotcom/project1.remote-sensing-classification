"""
FusionCropNetV5 单元测试套件
"""

import pytest
import torch
import sys
sys.path.insert(0, '.')

class TestFiLM:
    """FiLM 调制单元测试"""
    
    def test_film_initialization(self):
        from models.fusion_net_v5 import FiLM
        
        film = FiLM(cond_ch=128, feat_ch=512)
        assert film.gamma_clip == 2.0
        assert film.beta_clip == 1.0
        assert film.gamma.weight.shape == (512, 128, 1, 1)
        assert film.beta.weight.shape == (512, 128, 1, 1)
    
    def test_film_forward(self):
        from models.fusion_net_v5 import FiLM
        
        film = FiLM(cond_ch=128, feat_ch=512)
        feat = torch.randn(2, 512, 16, 16)
        cond = torch.randn(2, 128, 16, 16)
        
        output = film(feat, cond)
        assert output.shape == feat.shape
        
        gamma = film.gamma(cond)
        beta = film.beta(cond)
        assert gamma.max() <= 2.0
        assert gamma.min() >= -2.0
        assert beta.max() <= 1.0
        assert beta.min() >= -1.0

class TestTemporalEncoderStream:
    """时序编码器单元测试"""
    
    def test_temporal_encoder_initialization(self):
        from models.fusion_net_v5 import TemporalEncoderStream
        
        encoder = TemporalEncoderStream(d_model=512, n_heads=8, n_layers=2)
        assert encoder.d_model == 512
        assert encoder.cls.shape == (1, 1, 512)
        assert encoder.fallback.shape == (1, 512)
    
    def test_temporal_encoder_forward(self):
        from models.fusion_net_v5 import TemporalEncoderStream
        
        encoder = TemporalEncoderStream(d_model=512, n_heads=8, n_layers=2)
        x = torch.randn(128, 12, 512)
        doy = torch.rand(128, 12)
        
        cls_out, seq_out = encoder(x, doy)
        assert cls_out.shape == (128, 512)
        assert seq_out.shape == (128, 12, 512)
    
    def test_temporal_encoder_with_mask(self):
        from models.fusion_net_v5 import TemporalEncoderStream
        
        encoder = TemporalEncoderStream(d_model=512, n_heads=8, n_layers=2)
        x = torch.randn(32, 12, 512)
        doy = torch.rand(32, 12)
        cloud_mask = torch.randint(0, 2, (32, 12)).bool()
        valid_count = torch.randint(1, 13, (32,))
        
        cls_out, seq_out = encoder(x, doy, cloud_mask=cloud_mask, valid_count=valid_count)
        assert cls_out.shape == (32, 512)
        assert seq_out.shape == (32, 12, 512)

class TestCrossModalAttention:
    """跨模态注意力单元测试"""
    
    def test_cross_modal_attention_initialization(self):
        from models.fusion_net_v5 import CrossModalAttention
        
        attention = CrossModalAttention(d_model=512, n_heads=8)
        assert attention.proj[0].in_channels == 1024
        assert attention.proj[0].out_channels == 512
    
    def test_cross_modal_attention_forward(self):
        from models.fusion_net_v5 import CrossModalAttention
        
        attention = CrossModalAttention(d_model=512, n_heads=8)
        opt = torch.randn(2, 512, 16, 16)
        sar = torch.randn(2, 512, 16, 16)
        
        output = attention(opt, sar)
        assert output.shape == opt.shape

class TestDEMSpatialConditioner:
    """DEM空间条件调制单元测试"""
    
    def test_dem_conditioner_initialization(self):
        from models.fusion_net_v5 import DEMSpatialConditioner
        
        conditioner = DEMSpatialConditioner(feat_ch=512, dem_ch=128)
        assert conditioner.film.gamma_clip == 2.0
    
    def test_dem_conditioner_forward(self):
        from models.fusion_net_v5 import DEMSpatialConditioner
        
        conditioner = DEMSpatialConditioner(feat_ch=512, dem_ch=128)
        fused = torch.randn(2, 512, 16, 16)
        dem = torch.randn(2, 128, 16, 16)
        
        output = conditioner(fused, dem)
        assert output.shape == fused.shape

class TestLateFusion:
    """后期融合单元测试"""
    
    def test_late_fusion_initialization(self):
        from models.fusion_net_v5 import LateFusion
        
        fusion = LateFusion(d_model=512)
        assert fusion.proj[0].in_features == 1024
        assert fusion.proj[0].out_features == 512
    
    def test_late_fusion_forward(self):
        from models.fusion_net_v5 import LateFusion
        
        fusion = LateFusion(d_model=512)
        xm = torch.randn(512, 512)
        opt = torch.randn(512, 512)
        sar = torch.randn(512, 512)
        
        output = fusion(xm, opt, sar)
        assert output.shape == xm.shape

class TestDecoder:
    """Decoder unit tests — Decoder outputs pre-head features (64ch)."""

    def test_decoder_initialization(self):
        from models._base import Decoder, CARAFEUp

        # default (use_carafe=True) with CARAFE projection
        decoder = Decoder(feat_dim=512, sar_ch_list=[64, 128])
        assert decoder.pre_head_ch == 64

        # use_carafe=False keeps ConvTranspose2d
        decoder_ct = Decoder(feat_dim=512, sar_ch_list=[64, 128], use_carafe=False)
        assert decoder_ct.pre_head_ch == 64
        assert hasattr(decoder_ct.up1, 'out_channels')

    def test_decoder_forward(self):
        from models._base import Decoder

        decoder = Decoder(feat_dim=512, sar_ch_list=[64, 128])
        final = torch.randn(2, 512, 16, 16)
        opt_p2 = torch.randn(2, 256, 32, 32)
        sar_s1 = torch.randn(2, 64, 64, 64)
        sar_s2 = torch.randn(2, 128, 32, 32)

        features = decoder(final, (opt_p2,), (sar_s1, sar_s2), (64, 64))
        assert features.shape == (2, 64, 64, 64)

class TestFusionCropNetV5:
    """完整模型单元测试"""
    
    def test_model_initialization(self):
        from models.fusion_net_v5 import FusionCropNetV5
        
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
    
    def test_model_forward_inference(self):
        from models.fusion_net_v5 import FusionCropNetV5
        
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        model.eval()
        
        opt_seq = torch.randn(1, 12, 10, 32, 32)
        sar_seq = torch.randn(1, 12, 5, 32, 32)
        dem = torch.randn(1, 5, 32, 32)
        doy = torch.rand(1, 12)
        
        with torch.no_grad():
            logits = model(opt_seq, sar_seq, dem, doy)
        
        assert logits.shape == (1, 7, 32, 32)
        preds = logits.argmax(dim=1)
        assert preds.min() >= 0
        assert preds.max() < 7
    
    def test_model_forward_training(self):
        from models.fusion_net_v5 import FusionCropNetV5
        
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        model.train()
        
        opt_seq = torch.randn(1, 12, 10, 32, 32)
        sar_seq = torch.randn(1, 12, 5, 32, 32)
        dem = torch.randn(1, 5, 32, 32)
        doy = torch.rand(1, 12)
        cloud_mask = torch.randint(0, 2, (1, 12, 32, 32)).float()
        valid_count = torch.randint(1, 13, (1, 32, 32))
        
        logits, ndvi_pred, consistency_loss = model(
            opt_seq, sar_seq, dem, doy, cloud_mask, valid_count
        )
        
        assert logits.shape == (1, 7, 32, 32)
        assert ndvi_pred is not None
        assert consistency_loss is not None
    
    def test_model_gradients(self):
        from models.fusion_net_v5 import FusionCropNetV5
        
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        
        opt_seq = torch.randn(1, 6, 10, 16, 16)
        sar_seq = torch.randn(1, 6, 5, 16, 16)
        dem = torch.randn(1, 5, 16, 16)
        doy = torch.rand(1, 6)
        labels = torch.randint(0, 7, (1, 16, 16))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        output = model(opt_seq, sar_seq, dem, doy)
        logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0

class TestLossFunctions:
    """损失函数单元测试"""
    
    def test_huber_loss(self):
        ndvi_pred = torch.randn(100)
        ndvi_tgt = torch.randn(100)
        
        loss = torch.nn.HuberLoss(delta=0.1)(ndvi_pred, ndvi_tgt)
        assert loss.item() >= 0
    
    def test_cross_entropy_loss(self):
        logits = torch.randn(2, 7, 32, 32)
        labels = torch.randint(0, 7, (2, 32, 32))

        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        assert loss.item() >= 0


class TestV5ProBackbones:
    """Smoke tests for pluggable backbones (Task 1) and CARAFE upsampler (Task 2)."""

    @pytest.mark.parametrize("backbone", [
        "resnet50", "convnext_tiny", "efficientnet_b0",
    ])
    def test_optical_encoder_multi_backbone(self, backbone):
        from models._base import OpticalEncoder, _BACKBONE_CHANNELS

        bb_ch = _BACKBONE_CHANNELS[backbone]
        enc = OpticalEncoder(10, 512, backbone, pretrained=False)
        m, p2, p3 = enc(torch.randn(2, 10, 224, 224))
        # Verify output shapes are consistent
        assert m.shape[0] == 2 and m.shape[1] == 512
        assert p2.shape[1] == 256
        assert p3.shape[1] == 256

    def test_carafe_up_smoke(self):
        from models._base import CARAFEUp

        cu = CARAFEUp(512, scale=2)
        y = cu(torch.randn(2, 512, 16, 16))
        assert y.shape == (2, 512, 32, 32)

    def test_decoder_carafe_path(self):
        from models._base import Decoder

        dec = Decoder(512, [64, 128], use_carafe=True)
        out = dec(
            torch.randn(2, 512, 16, 16),
            (torch.randn(2, 256, 32, 32),),
            (torch.randn(2, 64, 64, 64), torch.randn(2, 128, 32, 32)),
            (64, 64))
        assert out.shape == (2, 64, 64, 64)

    def test_decoder_convtranpose_path(self):
        from models._base import Decoder

        dec = Decoder(512, [64, 128], use_carafe=False)
        out = dec(
            torch.randn(2, 512, 16, 16),
            (torch.randn(2, 256, 32, 32),),
            (torch.randn(2, 64, 64, 64), torch.randn(2, 128, 32, 32)),
            (64, 64))
        assert out.shape == (2, 64, 64, 64)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
