# FusionCropNet model family
from .fusion_net import FusionCropNet, PretrainedWeightManager, CrossModalAttentionFusion
from .fusion_net_v4 import FusionCropNetV4, training_step as v4_training_step
from .fusion_net_v5 import FusionCropNetV5
from .fusion_net_v5_edl import (
    FusionCropNetV5EDL,
    EDLHead,
    EDLLoss,
    dirichlet_to_predictions,
    evidence_level_fusion,
    training_step,
)
from .fusion_net_v5pro import (
    FusionCropNetV5Pro,
    training_step as v5pro_training_step,
)
from .fusion_net_v6 import (
    FusionCropNetV6,
    v6_training_step,
)
from ._base import (
    DEMOpticalConditioner, CrossModalAttentionLight, ModalNormalize, DomainAdapter,
    ViTFeaturePyramid, ThreeExpertLateFusion,
    list_vit_foundation_models, load_vit_foundation_weights,
)
from .mil_module import (
    FusionCropNetV5MIL,
    MILAttentionPooling,
    GatedMILPooling,
    create_mil_model,
)
from .unet_transformer import UNetTransformer, UNetTransformerWithSAR

# Shared components
from .temporal import (
    TemporalEncoderStream,
    FourierDOYEncoding,
    ObsQualityToken,
    LateFusion as TemporalLateFusion,
)
from .temporal_lite import TemporalLite
from .heads import (
    SpatialRefinement,
    PhenologyAuxHead,
    UncertaintyHead,
    SWBlock,
)
from .dem_encoder import DEMEncoder, ThreeWayFusion

__all__ = [
    # V1
    "FusionCropNet",
    "PretrainedWeightManager",
    "CrossModalAttentionFusion",
    # V4
    "FusionCropNetV4",
    "v4_training_step",
    # V5
    "FusionCropNetV5",
    # V5 EDL
    "FusionCropNetV5EDL",
    "EDLHead",
    "EDLLoss",
    "dirichlet_to_predictions",
    "evidence_level_fusion",
    "training_step",
    # V5 Pro
    "FusionCropNetV5Pro",
    "v5pro_training_step",
    # V6
    "FusionCropNetV6",
    "v6_training_step",
    "DEMOpticalConditioner",
    "CrossModalAttentionLight",
    "ModalNormalize",
    # MIL
    "FusionCropNetV5MIL",
    "MILAttentionPooling",
    "GatedMILPooling",
    "create_mil_model",
    # UNet
    "UNetTransformer",
    "UNetTransformerWithSAR",
    # Shared
    "TemporalEncoderStream",
    "FourierDOYEncoding",
    "ObsQualityToken",
    "TemporalLateFusion",
    "TemporalLite",
    "SpatialRefinement",
    "PhenologyAuxHead",
    "UncertaintyHead",
    "SWBlock",
    "DEMEncoder",
    "ThreeWayFusion",
    "DomainAdapter",
    # V6 Block 9 P3
    "ViTFeaturePyramid",
    "ThreeExpertLateFusion",
    "list_vit_foundation_models",
    "load_vit_foundation_weights",
]
