# Critical Fixes Summary

## Issues Fixed

### 1. **Reproducibility: Random Seed Not Set in Download Pipeline**
**File**: `src/data/download.py`
**Issue**: The downloader used `random.sample()` without setting a seed, making class balancing non-deterministic.
**Fix**:
- Added `seed` parameter to `FlipkartDataDownloader.__init__`
- Set `random.seed(seed)` and `np.random.seed(seed)` in __init__
- Removed inline `import random` from `balance_classes()` method
- Ensured filename generation uses stable counter instead of DataFrame index

**Impact**: Dataset downloads are now reproducible across runs.

### 2. **Test Set Leakage: Evaluation Created New Split**
**File**: `src/evaluate/evaluator.py`
**Issue**: The evaluator's `main()` function called `get_data_loaders()` which created a NEW train/val/test split, potentially using test data during training or vice versa.
**Fix**:
- Added `use_saved_split=True` parameter to evaluator's data loading
- Modified `evaluate()` to accept optional `val_loader` for threshold computation
- Changed threshold computation to use validation set, not test set
- Added class mapping verification warning if mismatch detected

**Impact**: Test set is now truly held out and never used during model selection.

### 3. **Label Mapping Inconsistency: Sorted Class Names**
**File**: `src/data/dataset.py`
**Issue**: Class names were sorted alphabetically when loading dataset, which could cause different `class_to_idx` mappings if directory order changed.
**Fix**:
- Modified `ProductImageDataset` to load `class_to_idx` from saved metadata if available
- Updated `download.py` to save `class_to_idx` in metadata (sorted for consistency)
- Ensures consistent label mapping across all runs

**Impact**: Class indices are now deterministic and consistent across train/eval/demo.

### 4. **Split Indices Not Saved: No Test Set Isolation**
**File**: `src/data/dataset.py`
**Issue**: Stratified split indices were generated but not saved, so each run could get different test sets.
**Fix**:
- Added `use_saved_split` parameter to `get_data_loaders()`
- Save split indices to `data/processed/split_indices.json` on first run
- Load saved indices on subsequent runs if `use_saved_split=True`
- Verify saved indices match dataset size before using
- Added tests to verify no overlap between train/val/test sets

**Impact**: Test set is now consistent across runs and truly isolated.

### 5. **Filename Instability: DataFrame Index Used**
**File**: `src/data/download.py`
**Issue**: Image filenames used DataFrame index (`{class_name}_{idx}.jpg`), which could change if dataset ordering changed.
**Fix**:
- Changed to use stable counter (`{class_name}_{sample_idx:05d}.jpg`)
- Counter increments for each successfully downloaded image

**Impact**: Filenames are now stable and reproducible.

### 6. **Metadata Missing Seed**
**File**: `src/data/download.py`
**Issue**: Metadata didn't include the seed used, making reproducibility verification difficult.
**Fix**:
- Added `seed` to saved metadata
- Added `class_to_idx` to saved metadata

**Impact**: Reproducibility can be verified from metadata.

### 7. **Transform Consistency**
**File**: `app/demo.py`
**Issue**: Demo transforms weren't explicitly documented as matching training/eval transforms.
**Fix**:
- Added comment noting transforms match training/eval pipeline
- Added logging to confirm transform configuration

**Impact**: Clear documentation that demo uses same preprocessing as training.

## Test Coverage

### New Test File: `tests/test_critical_fixes.py`

Tests added to verify:
1. **`test_split_indices_saved()`**: Verifies split indices are saved and have no overlap
2. **`test_class_mapping_consistency()`**: Verifies class_to_idx is deterministic
3. **`test_data_loader_uses_saved_split()`**: Verifies data loaders reuse saved splits
4. **`test_metadata_has_seed()`**: Verifies metadata includes seed
5. **`test_evaluation_uses_test_set_only()`**: Verifies evaluator is configured correctly

## Verification

### Syntax Validation
All modified files compile without syntax errors:
- `src/data/download.py` ✓
- `src/data/dataset.py` ✓
- `src/train/trainer.py` ✓
- `src/evaluate/evaluator.py` ✓
- `app/demo.py` ✓
- `tests/test_critical_fixes.py` ✓

### Runtime Testing
Tests could not be executed due to missing dependencies (torch, numpy, etc.) in current environment. However:
- All Python files compile successfully
- Logic changes are minimal and focused
- Test file provides verification once dependencies are installed

## Changes Summary

**Files Modified**:
1. `src/data/download.py` - Seed handling, filename stability, metadata saving
2. `src/data/dataset.py` - Saved split loading, class mapping consistency
3. `src/train/trainer.py` - Use saved split
4. `src/evaluate/evaluator.py` - Use saved split, separate val/test usage
5. `app/demo.py` - Transform documentation
6. `tests/test_critical_fixes.py` - New test file

**Files Created**:
- `tests/test_critical_fixes.py` - Critical fixes verification

**Lines Changed**: ~80 lines modified/added across 6 files

## Backward Compatibility

The changes are backward compatible:
- `use_saved_split=True` is default but can be disabled
- Old metadata without `class_to_idx` will fall back to directory scanning
- Old checkpoints without seed in metadata will still load
- Demo transforms unchanged (just documented)

## Next Steps

To verify fixes work correctly:
1. Install dependencies: `pip install -r requirements.txt`
2. Run data download: `python -m src.data.download`
3. Run critical tests: `python tests/test_critical_fixes.py`
4. Run model tests: `python tests/test_model.py`
5. Train model: `python -m src.train.trainer`
6. Evaluate: `python -m src.evaluate.evaluator`
7. Verify test set is truly held out by checking evaluation uses saved split

## Notes

- No GPU training was executed (environment lacks PyTorch)
- No actual metrics were generated or claimed
- All changes are code-level fixes based on code inspection
- Syntax validation passed for all modified files
