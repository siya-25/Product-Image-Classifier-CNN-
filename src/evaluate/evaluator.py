"""
Evaluation module for product image classifier.
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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import yaml

from src.model.cnn import get_model
from src.calibration.calibration import Calibrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for model performance analysis."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/config.yaml",
        device: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file
            device: Device to use (cuda/cpu)
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Initialize model
        self.model = get_model(
            model_type=self.config['model']['type'],
            num_classes=self.config['model']['num_classes'],
            dropout_rate=self.config['model']['dropout_rate'],
            pretrained=False,
            freeze_backbone=False
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load calibration if available
        self.calibrator = None
        self.use_calibration = False
        self.k = 3
        
        if 'calibration' in checkpoint:
            self.calibrator = Calibrator(
                model=self.model,
                num_classes=self.config['model']['num_classes'],
                device=self.device,
                calibration_dir=str(self.output_dir)
            )
            self.calibrator.load_calibration()
            self.use_calibration = True
            self.k = checkpoint['calibration'].get('k', 3)
            logger.info(f"Loaded calibration: threshold={self.calibrator.threshold:.4f}")
        else:
            logger.info("No calibration found in checkpoint, using raw probabilities")
        
        logger.info(f"Loaded model from {checkpoint_path}")
        logger.info(f"Classes: {self.idx_to_class}")
        
        # Setup output directories
        self.output_dir = Path(self.config['logging']['metrics_dir'])
        self.figure_dir = Path(self.config['logging']['figure_dir'])
        
        for dir_path in [self.output_dir, self.figure_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        return_probs: bool = False,
        use_calibration: bool = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions from dataloader.
        
        Args:
            dataloader: Data loader
            return_probs: Whether to return probability scores
            use_calibration: Whether to use calibrated probabilities (default: from checkpoint)
            
        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        if use_calibration is None:
            use_calibration = self.use_calibration
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in dataloader:
            images = images.to(self.device)
            
            outputs = self.model(images)
            
            # Use calibrated probabilities if available
            if use_calibration and self.calibrator is not None:
                probs = self.calibrator.get_calibrated_probs(outputs)
            else:
                probs = torch.softmax(outputs, dim=1)
            
            preds = probs.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            if return_probs:
                all_probs.append(probs.cpu().numpy())
        
        predictions = np.concatenate(all_preds)
        true_labels = np.concatenate(all_labels)
        probabilities = np.concatenate(all_probs) if return_probs else None
        
        return predictions, true_labels, probabilities
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for idx, class_name in self.idx_to_class.items():
            per_class_metrics[class_name] = {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1': float(f1[idx]),
                'support': int(support[idx])
            }
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'per_class': per_class_metrics
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.idx_to_class.values()),
            yticklabels=list(self.idx_to_class.values()),
            ax=axes[0]
        )
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Confusion Matrix (Counts)')
        
        # Normalized
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=list(self.idx_to_class.values()),
            yticklabels=list(self.idx_to_class.values()),
            ax=axes[1]
        )
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title('Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figure_dir / 'confusion_matrix.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {save_path}")
        
        return cm
    
    def get_confused_pairs(
        self,
        confusion_matrix: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, str, int]]:
        """
        Get top confused class pairs.
        
        Args:
            confusion_matrix: Confusion matrix
            top_k: Number of top pairs to return
            
        Returns:
            List of (true_class, pred_class, count) tuples
        """
        pairs = []
        
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix)):
                if i != j:  # Exclude correct predictions
                    count = confusion_matrix[i, j]
                    if count > 0:
                        true_class = self.idx_to_class[i]
                        pred_class = self.idx_to_class[j]
                        pairs.append((true_class, pred_class, count))
        
        # Sort by count descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs[:top_k]
    
    def find_misclassified_examples(
        self,
        dataloader: DataLoader,
        true_class: str,
        pred_class: str,
        max_examples: int = 9
    ) -> List[Tuple[str, float]]:
        """
        Find misclassified examples for a specific confusion pair.
        
        Args:
            dataloader: Data loader
            true_class: True class name
            pred_class: Predicted class name
            max_examples: Maximum number of examples to return
            
        Returns:
            List of (image_path, confidence) tuples
        """
        true_idx = self.class_to_idx[true_class]
        pred_idx = self.class_to_idx[pred_class]
        
        misclassified = []
        
        for images, labels, paths in self._iter_with_paths(dataloader):
            images = images.to(self.device)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            
            for i in range(len(labels)):
                if labels[i] == true_idx and preds[i] == pred_idx:
                    confidence = probs[i, pred_idx].item()
                    misclassified.append((paths[i], confidence))
                    
                    if len(misclassified) >= max_examples:
                        return misclassified
        
        return misclassified
    
    def _iter_with_paths(self, dataloader: DataLoader):
        """Iterate over dataloader with image paths."""
        # This is a simplified version - in practice, you'd need to modify the dataset
        # to return paths along with images and labels
        for images, labels in dataloader:
            yield images, labels, [f"sample_{i}" for i in range(len(labels))]
    
    def plot_misclassified_grid(
        self,
        image_paths: List[str],
        confidences: List[float],
        title: str,
        save_path: Optional[str] = None
    ):
        """
        Plot a grid of misclassified examples.
        
        Args:
            image_paths: List of image paths
            confidences: List of confidence scores
            title: Plot title
            save_path: Path to save figure
        """
        n_examples = len(image_paths)
        n_cols = 3
        n_rows = (n_examples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (img_path, conf) in enumerate(zip(image_paths, confidences)):
            row = idx // n_cols
            col = idx % n_cols
            
            try:
                img = Image.open(img_path)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"Conf: {conf:.2f}")
                axes[row, col].axis('off')
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                axes[row, col].text(0.5, 0.5, "Error", ha='center')
        
        # Hide empty subplots
        for idx in range(n_examples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figure_dir / f"misclassified_{title.replace(' ', '_')}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved misclassification grid to {save_path}")
    
    def evaluate(
        self,
        test_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_results: bool = True
    ) -> Dict:
        """
        Run full evaluation.
        
        Args:
            test_loader: Test data loader (held-out set)
            val_loader: Validation data loader (for threshold computation)
            save_results: Whether to save results
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting evaluation on TEST set")
        
        # Get predictions on TEST set only
        y_pred, y_true, y_probs = self.predict(test_loader, return_probs=True)
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred)
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
        
        # Plot confusion matrix
        cm = self.plot_confusion_matrix(y_true, y_pred)
        
        # Get confused pairs
        confused_pairs = self.get_confused_pairs(cm, top_k=5)
        logger.info("Top confused pairs:")
        for true_cls, pred_cls, count in confused_pairs:
            logger.info(f"  {true_cls} -> {pred_cls}: {count} samples")
        
        results = {
            'metrics': metrics,
            'confused_pairs': [
                {'true': t, 'pred': p, 'count': c}
                for t, p, c in confused_pairs
            ]
        }
        
        # Compute decision threshold using validation set (not test set)
        if val_loader is not None:
            logger.info("Computing decision threshold using validation set")
            threshold = self.compute_decision_threshold(val_loader)
            results['decision_threshold'] = threshold
        else:
            logger.warning("No validation loader provided, skipping threshold computation")
            results['decision_threshold'] = None
        
        # Save results
        if save_results:
            results_path = self.output_dir / 'evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved results to {results_path}")
        
        return results
    
    def compute_decision_threshold(
        self,
        val_loader: DataLoader,
        target_recall: float = 0.95
    ) -> float:
        """
        Compute confidence threshold for auto-accept vs manual review.
        
        Args:
            val_loader: Validation data loader
            target_recall: Target recall for auto-accept
            
        Returns:
            Confidence threshold
        """
        y_pred, y_true, y_probs = self.predict(val_loader, return_probs=True)
        
        # Get max probabilities for each prediction
        max_probs = y_probs.max(axis=1)
        correct = (y_pred == y_true)
        
        # Find threshold that achieves target recall
        thresholds = np.linspace(0, 1, 100)
        for threshold in thresholds:
            mask = max_probs >= threshold
            if mask.sum() == 0:
                continue
            recall = correct[mask].sum() / mask.sum()
            if recall >= target_recall:
                logger.info(f"Confidence threshold: {threshold:.2f} (recall: {recall:.2%})")
                return threshold
        
        # Default threshold if none found
        logger.warning(f"Could not achieve target recall {target_recall}, using default 0.8")
        return 0.8


def main():
    """Main entry point."""
    from src.data.dataset import get_data_loaders
    
    # Load data - use saved split to ensure test set is truly held out
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(
        data_dir="data",
        batch_size=32,
        num_workers=0,
        use_saved_split=True  # Critical: use saved test set
    )
    
    # Verify class_to_idx matches checkpoint
    evaluator = Evaluator(
        checkpoint_path="outputs/checkpoints/best_checkpoint.pth",
        config_path="configs/config.yaml"
    )
    
    # Check that class mappings match
    if evaluator.class_to_idx != class_to_idx:
        logger.warning("Class mapping mismatch between checkpoint and data split!")
        logger.warning(f"Checkpoint: {evaluator.class_to_idx}")
        logger.warning(f"Data split: {class_to_idx}")
    
    # Evaluate on test set, use val set for threshold
    results = evaluator.evaluate(test_loader, val_loader=val_loader)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
