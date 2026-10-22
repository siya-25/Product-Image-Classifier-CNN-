"""
Training module for product image classifier.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.cnn import get_model, count_parameters
from src.data.dataset import get_data_loaders
from src.calibration.calibration import Calibrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for product image classification."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed
        self.seed = self.config.get('seed', 42)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Setup directories
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        self.figure_dir = Path(self.config['logging']['figure_dir'])
        self.metrics_dir = Path(self.config['logging']['metrics_dir'])
        
        for dir_path in [self.log_dir, self.checkpoint_dir, self.figure_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = get_model(
            model_type=self.config['model']['type'],
            num_classes=self.config['model']['num_classes'],
            dropout_rate=self.config['model']['dropout_rate'],
            pretrained=self.config['model']['pretrained'],
            freeze_backbone=self.config['model']['freeze_backbone']
        ).to(self.device)
        
        logger.info(f"Model: {self.config['model']['type']}")
        logger.info(f"Trainable parameters: {count_parameters(self.model):,}")
        
        # Setup data loaders (use saved split for test set consistency)
        self.train_loader, self.val_loader, self.test_loader, self.class_to_idx = get_data_loaders(
            data_dir=self.config['data']['data_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            image_size=self.config['data']['image_size'],
            train_split=self.config['data']['train_split'],
            val_split=self.config['data']['val_split'],
            test_split=self.config['data']['test_split'],
            seed=self.seed,
            use_saved_split=True
        )
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup scheduler
        scheduler_type = self.config['scheduler']['type']
        if scheduler_type == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config['scheduler']['factor'],
                patience=self.config['scheduler']['patience'],
                min_lr=self.config['scheduler']['min_lr']
            )
        elif scheduler_type == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=self.config['scheduler']['min_lr']
            )
        elif scheduler_type == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=self.config['scheduler']['factor']
            )
        else:
            self.scheduler = None
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Early stopping
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = self.config['training']['early_stopping_patience']
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip_value'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_value']
                )
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            if global_step % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/BatchAcc', 100.*correct/total, global_step)
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, loader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'class_to_idx': self.class_to_idx
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with val_acc: {self.best_val_acc:.2f}%")
        
        # Save periodic checkpoint
        if epoch % self.config['logging']['save_every'] == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, periodic_path)
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        save_path = self.figure_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves to {save_path}")
    
    def train(self):
        """Run full training loop."""
        logger.info("Starting training")
        logger.info(f"Config: {json.dumps(self.config, indent=2)}")
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(self.val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            logger.info(
                f"Epoch {epoch}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f}"
            )
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Plot training curves
        self.plot_training_curves()
        
        # Save history
        history_path = self.metrics_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training completed. Best val acc: {self.best_val_acc:.2f}%")
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.best_val_acc
    
    def calibrate_model(
        self,
        target_precision: float = 0.95,
        k: int = 3
    ) -> Dict:
        """
        Calibrate model probabilities and select confidence threshold.
        
        Args:
            target_precision: Target precision for auto-accept
            k: Number of top predictions to return
            
        Returns:
            Calibration results
        """
        logger.info("Starting model calibration")
        
        # Load best checkpoint
        best_checkpoint = self.checkpoint_dir / 'best_checkpoint.pth'
        if not best_checkpoint.exists():
            logger.warning("Best checkpoint not found, skipping calibration")
            return {}
        
        # Initialize calibrator
        calibrator = Calibrator(
            model=self.model,
            num_classes=self.config['model']['num_classes'],
            device=self.device,
            calibration_dir=str(self.checkpoint_dir)
        )
        
        # Load best model state
        checkpoint = torch.load(best_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Calibrate on validation set
        calibration_results = calibrator.calibrate(
            self.val_loader,
            target_precision=target_precision,
            k=k
        )
        
        # Save calibration results to checkpoint
        checkpoint['calibration'] = calibration_results
        torch.save(checkpoint, best_checkpoint)
        
        logger.info(f"Calibration complete: threshold={calibration_results['threshold']:.4f}")
        
        return calibration_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate', action='store_true', help='Run calibration after training')
    parser.add_argument('--target_precision', type=float, default=0.95, help='Target precision for calibration')
    parser.add_argument('--k', type=int, default=3, help='Number of top predictions')
    args = parser.parse_args()
    
    trainer = Trainer(config_path="configs/config.yaml")
    best_acc = trainer.train()
    print(f"\nBest validation accuracy: {best_acc:.2f}%")
    
    if args.calibrate:
        print("\nRunning probability calibration...")
        calibration_results = trainer.calibrate_model(
            target_precision=args.target_precision,
            k=args.k
        )
        print(f"Calibration complete: threshold={calibration_results.get('threshold', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
