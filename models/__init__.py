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

# ── V6 Mathematical Theory Modules ──
from .geometric_invariants import (
    GeometricInvariantEncoder,
    geometric_invariant_stack,
    compute_geometric_invariants,
    darboux_frame,
    verify_se3_invariance,
)
from .grassmann_ot import (
    JointOTGrassmannAligner,
    sliced_gw_distance,
    grassmann_geodesic_distance,
    stiefel_admm_solver,
    geometric_anchor_distance,
)
from .siren_tta import (
    SIRENTTALoss,
    NTKStabilityMonitor,
    LRSAAdapter,
    HMAAdapter,
    TTAEngine,
)
from .topological_evidence import (
    TopologicalConflictClassifier,
    dirichlet_to_chain,
    cup_product,
    cohomology_conflict_detector,
    persistent_correction,
)

# ── V6 Math: P2+P3 Modules ──
from .siren_dem_encoder import SIRENDEMEncoder, SIRENLayer
from .tta_safety_monitor import TTASafetyMonitor, TopoEWC
from .synergy import (
    persistence_barcode,
    persistence_threshold_from_barcode,
    compute_parameter_persistence_importance,
    gradient_alignment,
    AdaptiveRegularizationScheduler,
)

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
    # V6 Math: Geometric Invariants
    "GeometricInvariantEncoder",
    "geometric_invariant_stack",
    "compute_geometric_invariants",
    "darboux_frame",
    "verify_se3_invariance",
    # V6 Math: Optimal Transport
    "JointOTGrassmannAligner",
    "sliced_gw_distance",
    "grassmann_geodesic_distance",
    "stiefel_admm_solver",
    "geometric_anchor_distance",
    # V6 Math: TTA
    "SIRENTTALoss",
    "NTKStabilityMonitor",
    "LRSAAdapter",
    "HMAAdapter",
    "TTAEngine",
    # V6 Math: Topological Evidence
    "TopologicalConflictClassifier",
    "dirichlet_to_chain",
    "cup_product",
    "cohomology_conflict_detector",
    "persistent_correction",
    # V6 Math: P2+P3
    "SIRENDEMEncoder",
    "SIRENLayer",
    "TTASafetyMonitor",
    "TopoEWC",
    "persistence_barcode",
    "persistence_threshold_from_barcode",
    "compute_parameter_persistence_importance",
    "gradient_alignment",
    "AdaptiveRegularizationScheduler",
    # V6 Block 9 P3
    "ViTFeaturePyramid",
    "ThreeExpertLateFusion",
    "list_vit_foundation_models",
    "load_vit_foundation_weights",
]
