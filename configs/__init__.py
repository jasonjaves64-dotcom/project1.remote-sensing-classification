"""Configuration loader for FusionCropNet V6.

Loads configs/v6_production.yaml and provides typed access to all parameters.
"""
import yaml
from pathlib import Path
from typing import Any, Dict

_CONFIG_PATH = Path(__file__).parent / "v6_production.yaml"
_config_cache: Dict[str, Any] = {}


def load_config(path: str = None) -> Dict[str, Any]:
    """Load YAML configuration with caching.

    Args:
        path: optional override config path (default: configs/v6_production.yaml)

    Returns:
        dict with all configuration sections
    """
    global _config_cache
    config_path = Path(path) if path else _CONFIG_PATH

    cache_key = str(config_path)
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    _config_cache[cache_key] = config
    return config


def get_model_config(path: str = None) -> dict:
    """Get model section of config."""
    return load_config(path).get('model', {})


def get_math_config(path: str = None) -> dict:
    """Get mathematical theory section of config."""
    return load_config(path).get('math', {})


def get_training_config(path: str = None) -> dict:
    """Get training section of config."""
    return load_config(path).get('training', {})


def get_inference_config(path: str = None) -> dict:
    """Get inference section of config."""
    return load_config(path).get('inference', {})


def get_deployment_config(path: str = None) -> dict:
    """Get deployment section of config."""
    return load_config(path).get('deployment', {})
