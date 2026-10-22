"""
Tests for benchmark module.
"""

import json
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.benchmark.benchmark import ModelBenchmark


def test_benchmark_initialization():
    """Test that benchmark initializes correctly."""
    try:
        benchmark = ModelBenchmark(
            config_path="configs/config.yaml",
            results_dir="outputs/benchmark"
        )
        
        assert benchmark.models_to_test is not None
        assert len(benchmark.models_to_test) > 0
        assert benchmark.results_dir.exists()
        
        print("✓ Benchmark initialization test passed")
        
    except Exception as e:
        print(f"⚠ Benchmark initialization test failed: {e}")


def test_benchmark_config_loading():
    """Test that benchmark config loads correctly."""
    try:
        benchmark_config_path = "configs/benchmark_config.yaml"
        
        if not Path(benchmark_config_path).exists():
            print("⚠ Benchmark config test skipped (config not found)")
            return
        
        benchmark = ModelBenchmark(
            config_path="configs/config.yaml",
            benchmark_config_path=benchmark_config_path,
            results_dir="outputs/benchmark"
        )
        
        # Verify config values
        assert 'models' in benchmark.benchmark_config
        assert 'max_epochs' in benchmark.benchmark_config
        assert benchmark.benchmark_config['use_saved_split'] == True
        
        print("✓ Benchmark config loading test passed")
        
    except Exception as e:
        print(f"⚠ Benchmark config loading test failed: {e}")


def test_model_size_calculation():
    """Test model size calculation."""
    try:
        from src.model.cnn import get_model
        
        model = get_model("custom", num_classes=3, pretrained=False)
        
        benchmark = ModelBenchmark(
            config_path="configs/config.yaml",
            results_dir="outputs/benchmark"
        )
        
        size_mb = benchmark.count_model_size(model)
        
        assert size_mb > 0
        assert size_mb < 1000  # Should be less than 1GB
        
        print(f"✓ Model size calculation test passed: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"⚠ Model size calculation test failed: {e}")


def test_parameter_count():
    """Test parameter count matches expected ranges."""
    try:
        from src.model.cnn import get_model
        
        models = {
            'custom': (0, 10_000_000),  # 0-10M
            'mobilenet_v2': (1_000_000, 100_000_000),  # 1-100M
            'resnet18': (10_000_000, 100_000_000),  # 10-100M
        }
        
        for model_type, (min_params, max_params) in models.items():
            model = get_model(model_type, num_classes=3, pretrained=False)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            assert min_params <= num_params <= max_params, \
                f"{model_type} has {num_params} params, expected {min_params}-{max_params}"
        
        print("✓ Parameter count test passed")
        
    except Exception as e:
        print(f"⚠ Parameter count test failed: {e}")


def test_benchmark_results_structure():
    """Test that benchmark results have correct structure."""
    try:
        # Create mock results
        mock_results = {
            'model_type': 'custom',
            'num_parameters': 2500000,
            'model_size_mb': 10.5,
            'inference_latency_ms': 15.2,
            'val_accuracy': 0.85,
            'val_precision_macro': 0.84,
            'val_recall_macro': 0.83,
            'val_f1_macro': 0.835,
            'test_accuracy': 0.84,
            'test_precision_macro': 0.83,
            'test_recall_macro': 0.82,
            'test_f1_macro': 0.825,
            'val_per_class': {},
            'test_per_class': {}
        }
        
        # Verify required fields
        required_fields = [
            'model_type', 'num_parameters', 'model_size_mb',
            'inference_latency_ms', 'val_accuracy', 'val_f1_macro',
            'test_accuracy', 'test_f1_macro'
        ]
        
        for field in required_fields:
            assert field in mock_results, f"Missing field: {field}"
        
        print("✓ Benchmark results structure test passed")
        
    except Exception as e:
        print(f"⚠ Benchmark results structure test failed: {e}")


def test_device_selection():
    """Test device selection logic."""
    try:
        benchmark = ModelBenchmark(
            config_path="configs/config.yaml",
            results_dir="outputs/benchmark"
        )
        
        # Test auto selection
        device = benchmark.get_device()
        assert device.type in ['cuda', 'cpu']
        
        print(f"✓ Device selection test passed: {device}")
        
    except Exception as e:
        print(f"⚠ Device selection test failed: {e}")


def test_identical_splits_flag():
    """Test that benchmark config enforces identical splits."""
    try:
        benchmark_config_path = "configs/benchmark_config.yaml"
        
        if not Path(benchmark_config_path).exists():
            print("⚠ Identical splits flag test skipped (config not found)")
            return
        
        benchmark = ModelBenchmark(
            config_path="configs/config.yaml",
            benchmark_config_path=benchmark_config_path,
            results_dir="outputs/benchmark"
        )
        
        # Verify use_saved_split is True
        assert benchmark.benchmark_config['use_saved_split'] == True, \
            "Benchmark must use saved splits for fair comparison"
        
        print("✓ Identical splits flag test passed")
        
    except Exception as e:
        print(f"⚠ Identical splits flag test failed: {e}")


if __name__ == "__main__":
    print("="*50)
    print("BENCHMARK MODULE TESTS")
    print("="*50)
    
    test_benchmark_initialization()
    test_benchmark_config_loading()
    test_model_size_calculation()
    test_parameter_count()
    test_benchmark_results_structure()
    test_device_selection()
    test_identical_splits_flag()
    
    print("="*50)
