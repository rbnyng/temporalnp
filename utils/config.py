import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_config = _make_serializable(config)

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def _make_serializable(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def get_global_bounds(config: Dict[str, Any]) -> Optional[tuple]:
    if 'global_bounds' in config:
        return tuple(config['global_bounds'])
    return None
