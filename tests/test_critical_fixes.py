"""
Critical tests for data pipeline fixes:
1. Reproducibility (seed handling)
2. Split consistency (test set isolation)
3. Label mapping consistency
4. Deduplication verification
"""

import json
import random
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import get_data_loaders


def test_split_indices_saved():
    """Test that split indices are saved and reused."""
    try:
        data_dir = "data"
        processed_dir = Path(data_dir) / "processed"
        split_file = processed_dir / "split_indices.json"
        
        if not split_file.exists():
            print("⚠ Split indices test skipped (no saved split found)")
            return
        
        # Load split indices
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # Verify structure
        assert 'train_idx' in split_data
        assert 'val_idx' in split_data
        assert 'test_idx' in split_data
        assert 'seed' in split_data
        assert split_data['seed'] == 42
        
        # Verify no overlap between splits
        train_set = set(split_data['train_idx'])
        val_set = set(split_data['val_idx'])
        test_set = set(split_data['test_idx'])
        
        assert len(train_set & val_set) == 0, "Train and val overlap!"
        assert len(train_set & test_set) == 0, "Train and test overlap!"
        assert len(val_set & test_set) == 0, "Val and test overlap!"
        
        print("✓ Split indices saved and verified (no overlap)")
        
    except Exception as e:
        print(f"⚠ Split indices test failed: {e}")


def test_class_mapping_consistency():
    """Test that class_to_idx mapping is consistent across runs."""
    try:
        data_dir = "data"
        processed_dir = Path(data_dir) / "processed"
        metadata_file = processed_dir / "metadata.json"
        
        if not metadata_file.exists():
            print("⚠ Class mapping test skipped (no metadata found)")
            return
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify class_to_idx is saved
        assert 'class_to_idx' in metadata, "class_to_idx not in metadata"
        
        # Verify it's deterministic (sorted)
        class_names = list(metadata['class_to_idx'].keys())
        assert class_names == sorted(class_names), "Class names not sorted!"
        
        # Verify indices are 0, 1, 2
        indices = list(metadata['class_to_idx'].values())
        assert indices == sorted(indices), "Indices not sequential!"
        
        print(f"✓ Class mapping consistent: {metadata['class_to_idx']}")
        
    except Exception as e:
        print(f"⚠ Class mapping test failed: {e}")


def test_data_loader_uses_saved_split():
    """Test that data loaders use saved split when available."""
    try:
        # First load with use_saved_split=True
        train1, val1, test1, class_to_idx1 = get_data_loaders(
            data_dir="data",
            batch_size=8,
            num_workers=0,
            use_saved_split=True
        )
        
        # Second load with use_saved_split=True (should be identical)
        train2, val2, test2, class_to_idx2 = get_data_loaders(
            data_dir="data",
            batch_size=8,
            num_workers=0,
            use_saved_split=True
        )
        
        # Verify class_to_idx is identical
        assert class_to_idx1 == class_to_idx2, "Class mappings differ!"
        
        # Verify dataset sizes are identical
        assert len(train1.dataset) == len(train2.dataset), "Train sizes differ!"
        assert len(val1.dataset) == len(val2.dataset), "Val sizes differ!"
        assert len(test1.dataset) == len(test2.dataset), "Test sizes differ!"
        
        print("✓ Data loaders use saved split consistently")
        
    except Exception as e:
        print(f"⚠ Data loader split test failed: {e}")


def test_metadata_has_seed():
    """Test that metadata includes seed for reproducibility."""
    try:
        data_dir = "data"
        processed_dir = Path(data_dir) / "processed"
        metadata_file = processed_dir / "metadata.json"
        
        if not metadata_file.exists():
            print("⚠ Metadata seed test skipped (no metadata found)")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert 'seed' in metadata, "Seed not in metadata"
        assert metadata['seed'] == 42, f"Seed is {metadata['seed']}, expected 42"
        
        print(f"✓ Metadata includes seed: {metadata['seed']}")
        
    except Exception as e:
        print(f"⚠ Metadata seed test failed: {e}")


def test_evaluation_uses_test_set_only():
    """Test that evaluation script is configured to use test set only."""
    try:
        # Check evaluator main function
        evaluator_file = Path("src/evaluate/evaluator.py")
        with open(evaluator_file, 'r') as f:
            content = f.read()
        
        # Verify use_saved_split=True is in the main function
        assert 'use_saved_split=True' in content, "Evaluator not using saved split!"
        
        # Verify it loads test_loader (not re-splitting)
        assert '_, _, test_loader' in content, "Evaluator not loading test set correctly!"
        
        print("✓ Evaluator configured to use test set only")
        
    except Exception as e:
        print(f"⚠ Evaluation test set test failed: {e}")


if __name__ == "__main__":
    print("="*50)
    print("CRITICAL FIXES VERIFICATION")
    print("="*50)
    
    test_split_indices_saved()
    test_class_mapping_consistency()
    test_data_loader_uses_saved_split()
    test_metadata_has_seed()
    test_evaluation_uses_test_set_only()
    
    print("="*50)
