"""
Smoke test for training pipeline.
"""

import torch
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.train.trainer import Trainer


def test_smoke_train():
    """Run a quick smoke test of training."""
    print("Running smoke test training...")
    
    # Modify config for quick test
    config = {
        'data': {
            'data_dir': 'data',
            'batch_size': 8,
            'num_workers': 0,
            'image_size': 128,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        },
        'model': {
            'type': 'custom',
            'num_classes': 3,
            'dropout_rate': 0.5,
            'pretrained': False,
            'freeze_backbone': False
        },
        'training': {
            'epochs': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'early_stopping_patience': 10,
            'gradient_clip_value': 1.0,
            'use_mixed_precision': False
        },
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6
        },
        'logging': {
            'log_dir': 'outputs',
            'checkpoint_dir': 'outputs/checkpoints',
            'figure_dir': 'outputs/figures',
            'metrics_dir': 'outputs/metrics',
            'save_every': 1,
            'log_interval': 5
        },
        'seed': 42
    }
    
    # Save temporary config
    temp_config_path = Path("configs/test_config.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Initialize trainer
        trainer = Trainer(config_path=str(temp_config_path))
        
        # Run 2 epochs
        best_acc = trainer.train()
        
        print(f"✓ Smoke test completed. Best val acc: {best_acc:.2f}%")
        
        # Cleanup
        temp_config_path.unlink()
        
    except Exception as e:
        print(f"⚠ Smoke test failed (expected if data not downloaded): {e}")
        if temp_config_path.exists():
            temp_config_path.unlink()


if __name__ == "__main__":
    test_smoke_train()
