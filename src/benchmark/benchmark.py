"""
Model benchmarking module for comparing multiple architectures.
"""

import os
import json
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.cnn import get_model, count_parameters
from src.data.dataset import get_data_loaders
from src.train.trainer import Trainer
from src.evaluate.evaluator import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark multiple model architectures with identical data splits."""
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        benchmark_config_path: Optional[str] = None,
        results_dir: str = "outputs/benchmark"
    ):
        """
        Initialize benchmark.
        
        Args:
            config_path: Path to base training configuration
            benchmark_config_path: Path to benchmark-specific config
            results_dir: Directory to save benchmark results
        """
        # Load base config
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Load or create benchmark config
        if benchmark_config_path and Path(benchmark_config_path).exists():
            with open(benchmark_config_path, 'r') as f:
                self.benchmark_config = yaml.safe_load(f)
        else:
            self.benchmark_config = self._default_benchmark_config()
        
        # Setup directories
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Models to benchmark
        self.models_to_test = self.benchmark_config.get('models', [
            'custom',
            'mobilenet_v2',
            'resnet18',
            'efficientnet_b0'
        ])
        
        # Results storage
        self.results = []
        
        logger.info(f"Initialized benchmark for {len(self.models_to_test)} models")
        logger.info(f"Models: {self.models_to_test}")
    
    def _default_benchmark_config(self) -> Dict:
        """Default benchmark configuration."""
        return {
            'models': ['custom', 'mobilenet_v2', 'resnet18', 'efficientnet_b0'],
            'max_epochs': 20,  # Reduced for benchmarking
            'early_stopping_patience': 5,
            'batch_size': 32,
            'use_saved_split': True,  # Critical: use identical splits
            'skip_existing': True,  # Skip if checkpoint exists
            'device': 'auto'  # auto, cuda, cpu
        }
    
    def get_device(self) -> torch.device:
        """Get device for training."""
        device_str = self.benchmark_config.get('device', 'auto')
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device_str)
    
    def count_model_size(self, model: nn.Module) -> float:
        """
        Count model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def measure_inference_latency(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 10
    ) -> float:
        """
        Measure average inference latency per batch.
        
        Args:
            model: PyTorch model
            dataloader: Data loader
            device: Device to run on
            num_batches: Number of batches to measure
            
        Returns:
            Average latency in milliseconds
        """
        model.eval()
        latencies = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                images = images.to(device)
                
                # Warmup
                if i == 0:
                    _ = model(images)
                    continue
                
                # Measure
                start = time.perf_counter()
                _ = model(images)
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)  # Convert to ms
        
        return np.mean(latencies) if latencies else 0.0
    
    def train_model(
        self,
        model_type: str,
        device: torch.device
    ) -> Tuple[str, Dict]:
        """
        Train a single model.
        
        Args:
            model_type: Model architecture name
            device: Device to train on
            
        Returns:
            Tuple of (checkpoint_path, training_results)
        """
        logger.info(f"Training {model_type}")
        
        # Create model-specific config
        model_config = self.base_config.copy()
        model_config['model']['type'] = model_type
        model_config['model']['pretrained'] = True
        model_config['model']['freeze_backbone'] = False
        model_config['training']['epochs'] = self.benchmark_config['max_epochs']
        model_config['training']['early_stopping_patience'] = self.benchmark_config['early_stopping_patience']
        model_config['data']['batch_size'] = self.benchmark_config['batch_size']
        
        # Save model-specific config
        model_config_path = self.checkpoint_dir / f"{model_type}_config.yaml"
        with open(model_config_path, 'w') as f:
            yaml.dump(model_config, f)
        
        # Initialize trainer
        trainer = Trainer(config_path=str(model_config_path))
        
        # Override device if specified
        if device.type != 'auto':
            trainer.device = device
            trainer.model = trainer.model.to(device)
        
        # Train
        best_val_acc = trainer.train()
        
        # Save checkpoint with model-specific name
        checkpoint_path = self.checkpoint_dir / f"{model_type}_best.pth"
        original_checkpoint = self.checkpoint_dir.parent / "checkpoints" / "best_checkpoint.pth"
        
        if original_checkpoint.exists():
            import shutil
            shutil.copy(original_checkpoint, checkpoint_path)
        else:
            logger.warning(f"Original checkpoint not found at {original_checkpoint}")
            checkpoint_path = None
        
        # Get training history
        history_path = self.checkpoint_dir.parent / "metrics" / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = trainer.history
        
        training_results = {
            'best_val_acc': best_val_acc,
            'epochs_trained': len(history['train_loss']),
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
        }
        
        return str(checkpoint_path) if checkpoint_path else None, training_results
    
    def evaluate_model(
        self,
        checkpoint_path: str,
        model_type: str,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device
    ) -> Dict:
        """
        Evaluate a trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: Model architecture name
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to run on
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_type}")
        
        # Load model
        model = get_model(
            model_type=model_type,
            num_classes=self.base_config['model']['num_classes'],
            dropout_rate=self.base_config['model']['dropout_rate'],
            pretrained=False,
            freeze_backbone=False
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Count parameters
        num_params = count_parameters(model)
        
        # Count model size
        model_size_mb = self.count_model_size(model)
        
        # Measure inference latency on validation set
        inference_latency_ms = self.measure_inference_latency(
            model, val_loader, device, num_batches=10
        )
        
        # Evaluate on validation set
        evaluator = Evaluator(
            checkpoint_path=checkpoint_path,
            config_path=str(self.checkpoint_dir / f"{model_type}_config.yaml"),
            device=str(device)
        )
        
        val_results = evaluator.evaluate(val_loader, save_results=False)
        
        # Evaluate on test set (but only for final comparison, not model selection)
        test_results = evaluator.evaluate(test_loader, save_results=False)
        
        return {
            'model_type': model_type,
            'num_parameters': num_params,
            'model_size_mb': model_size_mb,
            'inference_latency_ms': inference_latency_ms,
            'val_accuracy': val_results['metrics']['accuracy'],
            'val_precision_macro': val_results['metrics']['precision_macro'],
            'val_recall_macro': val_results['metrics']['recall_macro'],
            'val_f1_macro': val_results['metrics']['f1_macro'],
            'test_accuracy': test_results['metrics']['accuracy'],
            'test_precision_macro': test_results['metrics']['precision_macro'],
            'test_recall_macro': test_results['metrics']['recall_macro'],
            'test_f1_macro': test_results['metrics']['f1_macro'],
            'val_per_class': val_results['metrics']['per_class'],
            'test_per_class': test_results['metrics']['per_class']
        }
    
    def run_benchmark(self):
        """Run full benchmark on all models."""
        logger.info("Starting model benchmark")
        
        # Setup device
        device = self.get_device()
        logger.info(f"Using device: {device}")
        
        # Load data loaders with saved split (identical for all models)
        train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(
            data_dir=self.base_config['data']['data_dir'],
            batch_size=self.benchmark_config['batch_size'],
            num_workers=0,  # Use 0 for compatibility
            image_size=self.base_config['data']['image_size'],
            train_split=self.base_config['data']['train_split'],
            val_split=self.base_config['data']['val_split'],
            test_split=self.base_config['data']['test_split'],
            seed=self.base_config['seed'],
            use_saved_split=True  # Critical: identical splits
        )
        
        logger.info(f"Data loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
        
        # Benchmark each model
        for model_type in self.models_to_test:
            checkpoint_path = self.checkpoint_dir / f"{model_type}_best.pth"
            
            # Skip if checkpoint exists and skip_existing is True
            if self.benchmark_config.get('skip_existing', True) and checkpoint_path.exists():
                logger.info(f"Skipping {model_type} (checkpoint exists)")
                # Load existing results if available
                results_path = self.checkpoint_dir / f"{model_type}_results.json"
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        model_results = json.load(f)
                    self.results.append(model_results)
                continue
            
            try:
                # Train model
                checkpoint_path, training_results = self.train_model(model_type, device)
                
                if checkpoint_path is None:
                    logger.error(f"Training failed for {model_type}, no checkpoint saved")
                    continue
                
                # Evaluate model
                eval_results = self.evaluate_model(
                    checkpoint_path, model_type, val_loader, test_loader, device
                )
                
                # Combine results
                model_results = {**training_results, **eval_results}
                self.results.append(model_results)
                
                # Save individual results
                results_path = self.checkpoint_dir / f"{model_type}_results.json"
                with open(results_path, 'w') as f:
                    json.dump(model_results, f, indent=2)
                
                logger.info(f"{model_type} completed - Val Acc: {model_results['val_accuracy']:.2%}")
                
            except Exception as e:
                logger.error(f"Failed to benchmark {model_type}: {e}")
                continue
        
        # Save combined results
        self.save_results()
        
        # Select best model based on validation F1 macro
        self.select_best_model()
        
        # Plot comparison chart
        self.plot_comparison()
        
        logger.info("Benchmark completed")
    
    def save_results(self):
        """Save combined benchmark results."""
        results_path = self.results_dir / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'models_tested': len(self.results),
                'results': self.results
            }, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
    
    def select_best_model(self):
        """Select best model based on validation F1 macro."""
        if not self.results:
            logger.warning("No results to select from")
            return
        
        # Sort by validation F1 macro
        sorted_results = sorted(self.results, key=lambda x: x['val_f1_macro'], reverse=True)
        best_model = sorted_results[0]
        
        selection = {
            'best_model': best_model['model_type'],
            'selection_metric': 'val_f1_macro',
            'selection_value': best_model['val_f1_macro'],
            'test_accuracy': best_model['test_accuracy'],
            'test_f1_macro': best_model['test_f1_macro'],
            'all_models_ranked': [
                {
                    'model': r['model_type'],
                    'val_f1_macro': r['val_f1_macro'],
                    'test_f1_macro': r['test_f1_macro']
                }
                for r in sorted_results
            ]
        }
        
        selection_path = self.results_dir / "model_selection.json"
        with open(selection_path, 'w') as f:
            json.dump(selection, f, indent=2)
        
        logger.info(f"Best model: {best_model['model_type']} (Val F1: {best_model['val_f1_macro']:.4f})")
    
    def plot_comparison(self):
        """Plot comparison chart of all models."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Prepare data
        model_names = [r['model_type'] for r in self.results]
        val_acc = [r['val_accuracy'] * 100 for r in self.results]
        val_f1 = [r['val_f1_macro'] * 100 for r in self.results]
        params = [r['num_parameters'] / 1e6 for r in self.results]  # In millions
        latency = [r['inference_latency_ms'] for r in self.results]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Validation accuracy
        axes[0, 0].bar(model_names, val_acc, color='steelblue')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Validation Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Validation F1 macro
        axes[0, 1].bar(model_names, val_f1, color='forestgreen')
        axes[0, 1].set_ylabel('F1 Macro (%)')
        axes[0, 1].set_title('Validation F1 Macro')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Parameters
        axes[1, 0].bar(model_names, params, color='coral')
        axes[1, 0].set_ylabel('Parameters (M)')
        axes[1, 0].set_title('Model Size (Parameters)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Inference latency
        axes[1, 1].bar(model_names, latency, color='purple')
        axes[1, 1].set_ylabel('Latency (ms/batch)')
        axes[1, 1].set_title('Inference Latency')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = self.results_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison chart to {plot_path}")


def main():
    """Main entry point."""
    benchmark = ModelBenchmark(
        config_path="configs/config.yaml",
        results_dir="outputs/benchmark"
    )
    
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
