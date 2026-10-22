"""
PyTorch Dataset class for product images.
"""

import os
import json
import copy
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class ProductImageDataset(Dataset):
    """Dataset for product image classification."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root data directory containing processed images
            split: One of 'train', 'val', 'test'
            transform: Image transformations
            class_to_idx: Mapping from class names to indices
        """
        self.data_dir = Path(data_dir) / "processed"
        self.split = split
        self.transform = transform
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Get class names - use saved mapping from metadata for consistency
        if class_to_idx is None:
            if 'class_to_idx' in self.metadata:
                # Use saved mapping for reproducibility
                self.class_to_idx = self.metadata['class_to_idx']
                self.classes = list(self.class_to_idx.keys())
            else:
                # Fallback to sorted directory names
                self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())
        
        # Load image paths and labels
        self.samples = []
        self.manifest_fingerprint = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        logger.info(f"Classes: {self.class_to_idx}")
    
    def _load_samples(self):
        """Load and validate the balanced sample manifest."""
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest is missing: {manifest_path}. "
                "Run the data download pipeline to generate it."
            )

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if manifest.get("class_to_idx") != self.class_to_idx:
            raise ValueError(f"Class mapping in {manifest_path} does not match metadata")

        manifest_samples = manifest.get("samples")
        if not isinstance(manifest_samples, list) or not manifest_samples:
            raise ValueError(f"Dataset manifest has no valid samples: {manifest_path}")

        sample_counts = {class_name: 0 for class_name in self.classes}
        for sample_idx, sample in enumerate(manifest_samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Invalid sample {sample_idx} in {manifest_path}")
            class_name = sample.get("label")
            sample_path = sample.get("path")
            if (
                not isinstance(class_name, str)
                or class_name not in self.class_to_idx
                or not isinstance(sample_path, str)
            ):
                raise ValueError(f"Invalid path or label for sample {sample_idx} in {manifest_path}")

            relative_path = Path(sample_path)
            if relative_path.is_absolute():
                raise ValueError(f"Manifest path must be relative: {sample_path}")
            image_path = (self.data_dir / relative_path).resolve()
            try:
                image_path.relative_to(self.data_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"Manifest path escapes the dataset directory: {sample_path}") from exc
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Manifest image is missing for sample {sample_idx}: {image_path}"
                )

            self.samples.append((str(image_path), self.class_to_idx[class_name]))
            sample_counts[class_name] += 1

        if manifest.get("class_counts") != sample_counts:
            raise ValueError(
                f"Manifest class counts are inconsistent: "
                f"declared {manifest.get('class_counts')}, found {sample_counts}"
            )
        if self.metadata.get("class_counts") != sample_counts:
            raise ValueError(
                f"Manifest class counts do not match metadata: {sample_counts}"
            )
        if len(set(sample_counts.values())) != 1:
            raise ValueError(f"Dataset manifest class counts are unbalanced: {sample_counts}")
        if self.metadata.get("total_images") != len(self.samples):
            raise ValueError(
                f"Manifest sample count {len(self.samples)} does not match metadata "
                f"total_images {self.metadata.get('total_images')}"
            )

        fingerprint = hashlib.sha256(
            json.dumps(manifest_samples, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError(f"Dataset manifest fingerprint is invalid: {manifest_path}")
        return fingerprint
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class."""
        distribution = {}
        for class_name in self.classes:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[class_name])
            distribution[class_name] = count
        return distribution


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    use_saved_split: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train, validation, and test data loaders with stratified split.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Image size for resizing
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        seed: Random seed for reproducibility
        use_saved_split: Whether to use saved split indices if available
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx)
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def make_eval_transform():
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transform = make_eval_transform()
    test_transform = make_eval_transform()
    
    # Load full dataset
    full_dataset = ProductImageDataset(
        data_dir=data_dir,
        split="full",
        transform=None
    )
    
    class_to_idx = full_dataset.class_to_idx
    
    # Check for saved split indices
    split_file = Path(data_dir) / "processed" / "split_indices.json"
    sample_fingerprint = full_dataset.manifest_fingerprint
    split_config = {
        'seed': seed,
        'train_split': train_split,
        'val_split': val_split,
        'test_split': test_split
    }
    train_idx = val_idx = test_idx = None
    
    if use_saved_split and split_file.exists():
        logger.info(f"Loading saved split indices from {split_file}")
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        saved_indices = [
            split_data.get('train_idx'),
            split_data.get('val_idx'),
            split_data.get('test_idx')
        ]
        valid_indices = (
            all(isinstance(indices, list) for indices in saved_indices)
            and all(type(index) is int for indices in saved_indices for index in indices)
            and sorted(index for indices in saved_indices for index in indices)
            == list(range(len(full_dataset)))
        )
        if (
            split_data.get('sample_fingerprint') == sample_fingerprint
            and split_data.get('total_samples') == len(full_dataset)
            and all(split_data.get(key) == value for key, value in split_config.items())
            and valid_indices
        ):
            train_idx, val_idx, test_idx = saved_indices
            logger.info("Using saved split indices for reproducibility")
        else:
            logger.warning("Saved split indices don't match current dataset or config, regenerating")
    
    if train_idx is None:
        # Stratified split
        indices = list(range(len(full_dataset)))
        labels = [label for _, label in full_dataset.samples]
        
        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_split + test_split),
            stratify=labels,
            random_state=seed
        )
        
        # Second split: val vs test
        temp_labels = [labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_split / (val_split + test_split),
            stratify=temp_labels,
            random_state=seed
        )
        
        # Save split indices for reproducibility
        split_data = {
            'total_samples': len(full_dataset),
            'sample_fingerprint': sample_fingerprint,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            **split_config
        }
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved split indices to {split_file}")
    
    # Create subset datasets
    train_data = copy.copy(full_dataset)
    val_data = copy.copy(full_dataset)
    test_data = copy.copy(full_dataset)
    train_data.split = "train"
    val_data.split = "val"
    test_data.split = "test"
    train_data.transform = train_transform
    val_data.transform = val_transform
    test_data.transform = test_transform
    
    train_dataset = torch.utils.data.Subset(train_data, train_idx)
    val_dataset = torch.utils.data.Subset(val_data, val_idx)
    test_dataset = torch.utils.data.Subset(test_data, test_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_to_idx


def main():
    """Test the dataset loader."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(
        data_dir="data",
        batch_size=32,
        num_workers=0,
        image_size=224
    )
    
    # Test batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Class to idx: {class_to_idx}")


if __name__ == "__main__":
    main()
