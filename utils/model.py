from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from models.neural_process import GEDINeuralProcess


def initialize_model(
    config: Dict[str, Any],
    device: str = 'cpu'
) -> GEDINeuralProcess:
    model = GEDINeuralProcess(
        patch_size=config.get('patch_size', 3),
        embedding_channels=128,
        embedding_feature_dim=config.get('embedding_feature_dim', 128),
        context_repr_dim=config.get('context_repr_dim', 128),
        hidden_dim=config.get('hidden_dim', 512),
        latent_dim=config.get('latent_dim', 128),
        output_uncertainty=True,
        architecture_mode=config.get('architecture_mode', 'deterministic'),
        num_attention_heads=config.get('num_attention_heads', 4)
    ).to(device)

    return model


def load_checkpoint(
    checkpoint_dir: Path,
    device: str = 'cpu',
    checkpoint_name: Optional[str] = None
) -> Tuple[Dict[str, Any], Path]:
    if checkpoint_name:
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    else:
        checkpoint_files = ['best_r2_model.pt', 'best_model.pt']
        checkpoint_path = None

        for ckpt_file in checkpoint_files:
            path = checkpoint_dir / ckpt_file
            if path.exists():
                checkpoint_path = path
                break

        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}. "
                f"Looked for: {checkpoint_files}"
            )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    return checkpoint, checkpoint_path


def load_model_from_checkpoint(
    checkpoint_dir: Path,
    device: str = 'cpu',
    checkpoint_name: Optional[str] = None
) -> Tuple[GEDINeuralProcess, Dict[str, Any], Path]:
    from .config import load_config

    config_path = checkpoint_dir / 'config.json'
    config = load_config(config_path)

    model = initialize_model(config, device)

    checkpoint, checkpoint_path = load_checkpoint(
        checkpoint_dir, device, checkpoint_name
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint, checkpoint_path
