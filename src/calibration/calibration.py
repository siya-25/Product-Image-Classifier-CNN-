"""
Probability calibration and confidence threshold selection.
Implements temperature scaling and precision-coverage analysis.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.
    
    Learns a single temperature parameter to calibrate model outputs.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model output logits (batch_size, num_classes)
            
        Returns:
            Calibrated probabilities (batch_size, num_classes)
        """
        return torch.softmax(logits / self.temperature, dim=1)
    
    def fit(
        self,
        val_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        max_epochs: int = 10,
        lr: float = 0.01
    ) -> float:
        """
        Fit temperature on validation set.
        
        Args:
            val_loader: Validation data loader
            model: Trained model
            device: Device to run on
            max_epochs: Maximum training epochs
            lr: Learning rate
            
        Returns:
            Final temperature value
        """
        self.to(device)
        self.train()
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr)
        
        criterion = nn.CrossEntropyLoss()
        
        temperature_history = []
        
        for epoch in range(max_epochs):
            def closure():
                optimizer.zero_grad()
                all_logits = []
                all_labels = []
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        logits = model(images)
                        all_logits.append(logits)
                        all_labels.append(labels)
                
                all_logits = torch.cat(all_logits)
                all_labels = torch.cat(all_labels).to(device)
                
                # Apply temperature scaling
                scaled_logits = all_logits / self.temperature
                loss = criterion(scaled_logits, all_labels)
                
                loss.backward()
                return loss
            
            optimizer.step(closure)
            temperature_history.append(self.temperature.item())
            
            # Check convergence
            if len(temperature_history) > 1:
                diff = abs(temperature_history[-1] - temperature_history[-2])
                if diff < 1e-4:
                    logger.info(f"Temperature converged at epoch {epoch}")
                    break
        
        self.eval()
        final_temp = self.temperature.item()
        logger.info(f"Temperature scaling complete: T={final_temp:.4f}")
        
        return final_temp


class Calibrator:
    """
    Calibrator for probability calibration and threshold selection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        device: torch.device,
        calibration_dir: str = "outputs/calibration"
    ):
        """
        Initialize calibrator.
        
        Args:
            model: Trained model
            num_classes: Number of classes
            device: Device to run on
            calibration_dir: Directory to save calibration data
        """
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        
        self.temperature_scaling = None
        self.temperature = None
        self.threshold = None
        self.calibration_config = {}
    
    def calibrate(
        self,
        val_loader: DataLoader,
        target_precision: float = 0.95,
        k: int = 3
    ) -> Dict:
        """
        Calibrate probabilities and select threshold.
        
        Args:
            val_loader: Validation data loader
            target_precision: Target precision for auto-accept
            k: Number of top predictions to return
            
        Returns:
            Calibration results
        """
        logger.info("Starting calibration")
        
        # Step 1: Temperature scaling
        logger.info("Step 1: Temperature scaling")
        self.temperature_scaling = TemperatureScaling()
        self.temperature = self.temperature_scaling.fit(
            val_loader, self.model, self.device
        )
        
        # Step 2: Precision-coverage analysis
        logger.info("Step 2: Precision-coverage analysis")
        precision_results = self.analyze_precision_coverage(
            val_loader, target_precision
        )
        
        # Step 3: Select threshold
        logger.info("Step 3: Selecting threshold")
        self.threshold = precision_results['threshold']
        
        # Save calibration results
        self.calibration_config = {
            'temperature': self.temperature,
            'threshold': self.threshold,
            'target_precision': target_precision,
            'k': k,
            'precision_at_threshold': precision_results['precision_at_threshold'],
            'coverage_at_threshold': precision_results['coverage_at_threshold']
        }
        
        self.save_calibration()
        
        logger.info(f"Calibration complete: T={self.temperature:.4f}, threshold={self.threshold:.4f}")
        
        return self.calibration_config
    
    def get_calibrated_probs(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Get calibrated probabilities.
        
        Args:
            logits: Model output logits
            
        Returns:
            Calibrated probabilities
        """
        if self.temperature_scaling is None:
            logger.warning("Temperature scaling not fitted, using raw probabilities")
            return torch.softmax(logits, dim=1)
        
        return self.temperature_scaling(logits)
    
    def analyze_precision_coverage(
        self,
        val_loader: DataLoader,
        target_precision: float = 0.95
    ) -> Dict:
        """
        Analyze precision vs coverage to select threshold.
        
        Args:
            val_loader: Validation data loader
            target_precision: Target precision for threshold selection
            
        Returns:
            Analysis results
        """
        self.model.eval()
        self.temperature_scaling.eval()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = self.get_calibrated_probs(logits)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # Get max probabilities and predictions
        max_probs = all_probs.max(axis=1)
        predictions = all_probs.argmax(axis=1)
        
        # Calculate precision at various thresholds
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        coverages = []
        
        for threshold in thresholds:
            mask = max_probs >= threshold
            if mask.sum() == 0:
                precisions.append(0.0)
                coverages.append(0.0)
                continue
            
            subset_preds = predictions[mask]
            subset_labels = all_labels[mask]
            
            precision = precision_score(subset_labels, subset_preds, average='macro', zero_division=0)
            coverage = mask.sum() / len(all_labels)
            
            precisions.append(precision)
            coverages.append(coverage)
        
        # Select threshold that meets target precision
        selected_threshold = 0.5  # Default
        precision_at_threshold = 0.0
        coverage_at_threshold = 0.0
        
        for threshold, precision, coverage in zip(thresholds, precisions, coverages):
            if precision >= target_precision:
                selected_threshold = threshold
                precision_at_threshold = precision
                coverage_at_threshold = coverage
                break
        
        # Plot precision-coverage curve
        self.plot_precision_coverage(thresholds, precisions, coverages, target_precision, selected_threshold)
        
        return {
            'thresholds': thresholds.tolist(),
            'precisions': precisions,
            'coverages': coverages,
            'threshold': selected_threshold,
            'precision_at_threshold': precision_at_threshold,
            'coverage_at_threshold': coverage_at_threshold
        }
    
    def plot_precision_coverage(
        self,
        thresholds: np.ndarray,
        precisions: List[float],
        coverages: List[float],
        target_precision: float,
        selected_threshold: float
    ):
        """
        Plot precision-coverage curve.
        
        Args:
            thresholds: Threshold values
            precisions: Precision at each threshold
            coverages: Coverage at each threshold
            target_precision: Target precision
            selected_threshold: Selected threshold
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax.plot(thresholds, coverages, label='Coverage', linewidth=2)
        ax.axhline(y=target_precision, color='r', linestyle='--', label=f'Target Precision ({target_precision})')
        ax.axvline(x=selected_threshold, color='g', linestyle='--', label=f'Selected Threshold ({selected_threshold:.2f})')
        
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision vs Coverage Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.calibration_dir / 'precision_coverage_curve.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved precision-coverage curve to {save_path}")
    
    def save_calibration(self):
        """Save calibration parameters."""
        config_path = self.calibration_dir / 'calibration_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.calibration_config, f, indent=2)
        
        # Save temperature scaling state
        if self.temperature_scaling is not None:
            temp_path = self.calibration_dir / 'temperature_scaling.pth'
            torch.save(self.temperature_scaling.state_dict(), temp_path)
            logger.info(f"Saved temperature scaling to {temp_path}")
        
        logger.info(f"Saved calibration config to {config_path}")
    
    def load_calibration(self, config_path: Optional[str] = None):
        """
        Load calibration parameters.
        
        Args:
            config_path: Path to calibration config (uses default if None)
        """
        if config_path is None:
            config_path = self.calibration_dir / 'calibration_config.json'
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Calibration config not found at {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            self.calibration_config = json.load(f)
        
        self.temperature = self.calibration_config['temperature']
        self.threshold = self.calibration_config['threshold']
        
        # Load temperature scaling
        temp_path = self.calibration_dir / 'temperature_scaling.pth'
        if temp_path.exists():
            self.temperature_scaling = TemperatureScaling()
            self.temperature_scaling.load_state_dict(torch.load(temp_path))
            self.temperature_scaling.eval()
            logger.info(f"Loaded temperature scaling from {temp_path}")
        
        logger.info(f"Loaded calibration: T={self.temperature:.4f}, threshold={self.threshold:.4f}")
        
        return True
    
    def predict_with_confidence(
        self,
        logits: torch.Tensor,
        k: int = 3
    ) -> Dict:
        """
        Predict with confidence scores and top-k predictions.
        
        Args:
            logits: Model output logits
            k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence
        """
        # Get calibrated probabilities
        probs = self.get_calibrated_probs(logits)
        
        # Get top-k predictions
        topk_probs, topk_indices = torch.topk(probs, k, dim=1)
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()
        topk_probs_np = topk_probs.cpu().numpy()
        topk_indices_np = topk_indices.cpu().numpy()
        
        # Get max prediction
        max_probs = probs_np.max(axis=1)
        max_indices = probs_np.argmax(axis=1)
        
        # Check if needs manual review
        needs_review = max_probs < self.threshold
        
        return {
            'predicted_class': int(max_indices[0]),
            'confidence': float(max_probs[0]),
            'needs_review': bool(needs_review[0]),
            'top_k_predictions': topk_indices[0].tolist(),
            'top_k_confidences': topk_probs_np[0].tolist(),
            'all_probabilities': probs_np[0].tolist()
        }
