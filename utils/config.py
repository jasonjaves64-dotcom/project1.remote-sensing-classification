from dataclasses import dataclass, field
from typing import Optional, Dict, List
import yaml
from pathlib import Path

@dataclass
class DataConfig:
    raw_dir: str = "./data/raw"
    landsat_dir: str = "./data/raw/landsat/2023"
    sar_dir: str = "./data/raw/sentinel1/2023"
    label_dir: str = "./data/raw/labels"
    processed_dir: str = "./data/processed"
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"

@dataclass
class ModelConfig:
    version: str = "v5"
    opt_channels: int = 10
    sar_channels: int = 5
    dem_channels: int = 5
    num_classes: int = 7
    feat_dim: int = 512
    backbone: str = "resnet50"
    pretrained: bool = True
    n_heads: int = 16
    win_size: int = 4
    n_layers: int = 4
    drop_timestep_p: float = 0.1

@dataclass
class EDLConfig:
    enabled: bool = True
    dropout_p: float = 0.3
    lambda_max: float = 0.5
    anneal_ep: int = 50
    n_passes: int = 5

@dataclass
class TrainingConfig:
    batch_size: int = 8
    patch_size: int = 32
    epochs_p1: int = 20
    epochs_p2: int = 60
    lr_p1: float = 0.001
    lr_p2: float = 0.0003
    weight_decay: float = 0.0001
    val_split: float = 0.15
    use_spatial_split: bool = False

@dataclass
class InferenceConfig:
    patch_size: int = 32
    overlap: float = 0.5
    batch_size: int = 16
    use_tta: bool = True

@dataclass
class VisualizationConfig:
    save_fig: bool = True
    fig_dpi: int = 150

@dataclass
class PreprocessingConfig:
    apply_terrain_correction: bool = False
    terrain_correction_method: str = "minnaert"
    dem_path: str = "./data/dem/dem_30m.tif"
    purity_threshold: float = 0.8
    split_block_size: int = 64
    max_cloud_pct: float = 0.5
    min_valid_observations: int = 4
    interpolation_method: str = "linear"
    normalization_strategy: str = "standard"
    spatial_split_kfold: int = 5

@dataclass
class GEEConfig:
    project_id: str = "your-gee-project-id"
    study_area: Dict[str, float] = field(default_factory=lambda: {
        "lon_min": 115.0, "lat_min": 36.0,
        "lon_max": 117.0, "lat_max": 38.0
    })
    year: int = 2023
    crs: str = "EPSG:32650"

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    edl: EDLConfig = field(default_factory=EDLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    gee: GEEConfig = field(default_factory=GEEConfig)
    crop_classes: Dict[int, str] = field(default_factory=lambda: {
        0: "背景", 1: "冬小麦", 2: "夏玉米",
        3: "水稻", 4: "大豆", 5: "棉花", 6: "其他作物"
    })

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """从YAML文件加载配置"""
        path = Path(path)
        if not path.exists():
            print(f"警告: 配置文件 {path} 不存在，使用默认配置")
            return cls()
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            edl=EDLConfig(**config_dict.get('edl', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
            gee=GEEConfig(**config_dict.get('gee', {})),
            crop_classes=config_dict.get('crop_classes', {
                0: "背景", 1: "冬小麦", 2: "夏玉米",
                3: "水稻", 4: "大豆", 5: "棉花", 6: "其他作物"
            })
        )

    def to_yaml(self, path: str):
        """将配置保存到YAML文件"""
        config_dict = {
            'data': {
                'raw_dir': self.data.raw_dir,
                'landsat_dir': self.data.landsat_dir,
                'sar_dir': self.data.sar_dir,
                'label_dir': self.data.label_dir,
                'processed_dir': self.data.processed_dir,
                'output_dir': self.data.output_dir,
                'checkpoint_dir': self.data.checkpoint_dir
            },
            'model': {
                'version': self.model.version,
                'opt_channels': self.model.opt_channels,
                'sar_channels': self.model.sar_channels,
                'dem_channels': self.model.dem_channels,
                'num_classes': self.model.num_classes,
                'feat_dim': self.model.feat_dim,
                'backbone': self.model.backbone,
                'pretrained': self.model.pretrained,
                'n_heads': self.model.n_heads,
                'win_size': self.model.win_size,
                'n_layers': self.model.n_layers,
                'drop_timestep_p': self.model.drop_timestep_p
            },
            'edl': {
                'enabled': self.edl.enabled,
                'dropout_p': self.edl.dropout_p,
                'lambda_max': self.edl.lambda_max,
                'anneal_ep': self.edl.anneal_ep,
                'n_passes': self.edl.n_passes
            },
            'training': {
                'batch_size': self.training.batch_size,
                'patch_size': self.training.patch_size,
                'epochs_p1': self.training.epochs_p1,
                'epochs_p2': self.training.epochs_p2,
                'lr_p1': self.training.lr_p1,
                'lr_p2': self.training.lr_p2,
                'weight_decay': self.training.weight_decay,
                'val_split': self.training.val_split,
                'use_spatial_split': self.training.use_spatial_split
            },
            'inference': {
                'patch_size': self.inference.patch_size,
                'overlap': self.inference.overlap,
                'batch_size': self.inference.batch_size,
                'use_tta': self.inference.use_tta
            },
            'visualization': {
                'save_fig': self.visualization.save_fig,
                'fig_dpi': self.visualization.fig_dpi
            },
            'preprocessing': {
                'apply_terrain_correction': self.preprocessing.apply_terrain_correction,
                'terrain_correction_method': self.preprocessing.terrain_correction_method,
                'dem_path': self.preprocessing.dem_path,
                'purity_threshold': self.preprocessing.purity_threshold,
                'split_block_size': self.preprocessing.split_block_size,
                'max_cloud_pct': self.preprocessing.max_cloud_pct,
                'min_valid_observations': self.preprocessing.min_valid_observations,
                'interpolation_method': self.preprocessing.interpolation_method,
                'normalization_strategy': self.preprocessing.normalization_strategy,
                'spatial_split_kfold': self.preprocessing.spatial_split_kfold
            },
            'gee': {
                'project_id': self.gee.project_id,
                'study_area': self.gee.study_area,
                'year': self.gee.year,
                'crs': self.gee.crs
            },
            'crop_classes': self.crop_classes
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def __repr__(self):
        return f"Config(data={self.data}, model={self.model}, edl={self.edl}, training={self.training})"