# Data Card

## Dataset Overview

This document describes the dataset used for training the Product Image Classifier.

## Dataset Information

### Source
- **Name**: PromptCloudHQ_flipkart-products
- **Provider**: Hugging Face (dataset by jason1966)
- **URL**: https://huggingface.co/datasets/jason1966/PromptCloudHQ_flipkart-products
- **Download Date**: 2026-10-10
- **Size**: ~20,000 products

### License
- **License Type**: Unknown
- **Restrictions**: Public dataset from Hugging Face
- **Commercial Use**: Verify before production deployment
- **Attribution**: Original data scraped from Flipkart by PromptCloud

### Data Fields
- `uniq_id`: Unique product identifier
- `product_category_tree`: Category hierarchy (JSON string)
- `image`: Image URLs (JSON string array)
- `product_name`: Product name
- `description`: Product description
- `brand`: Brand name
- `retail_price`: Original price
- `discounted_price`: Discounted price

## Processing Pipeline

### 1. Category Mapping
The raw Flipkart category tree is mapped to our 3 classes using keyword matching:

```python
# Example mapping
"Clothing >> Women's Clothing >> Lingerie" -> "Apparel"
"Furniture >> Living Room Furniture >> Sofa" -> "Home"
"Mobiles >> Smartphones >> Android" -> "Electronics"
```

Complete mapping in `configs/label_map.yaml`.

### 2. Image Download
- Images downloaded from URLs in the dataset
- Retry logic: 3 attempts with 10s timeout
- Resized to 224x224 pixels
- Converted to RGB if necessary
- Saved as JPEG with 85% quality

### 3. Deduplication
- Perceptual hashing (pHash) used to detect duplicates
- Duplicate images removed before train/val/test split
- Prevents data leakage between splits

### 4. Class Balancing
- Target: 1,500 images per class
- Maximum ratio: 2x between classes
- Random sampling if class exceeds target
- Ensures balanced training

### 5. Train/Val/Test Split
- Stratified split to maintain class distribution
- Train: 70%
- Validation: 15%
- Test: 15%
- Random seed: 42 (reproducible)

## Final Dataset Statistics

### Class Distribution
| Class | Training | Validation | Test | Total |
|-------|----------|------------|------|-------|
| Apparel | ~1,050 | ~225 | ~225 | ~1,500 |
| Electronics | ~1,050 | ~225 | ~225 | ~1,500 |
| Home | ~1,050 | ~225 | ~225 | ~1,500 |
| **Total** | **~3,150** | **~675** | **~675** | **~4,500** |

*Note: Actual numbers may vary based on download success rate and deduplication.*

### Image Specifications
- **Resolution**: 224x224 pixels
- **Color Space**: RGB
- **Format**: JPEG
- **Quality**: 85%
- **File Size**: ~20-50 KB per image

## Data Quality Issues

### Known Issues
1. **Broken URLs**: Some image URLs in the original dataset are no longer valid
2. **Image Quality**: Varies across products (some low-resolution, some watermarked)
3. **Category Noise**: Some products may be misclassified in the original dataset
4. **Background Clutter**: Product images may include background context

### Mitigation Strategies
- Retry logic for failed downloads
- Perceptual hash deduplication
- Data augmentation to improve robustness
- Error analysis to identify systematic issues

## Data Augmentation

### Training Augmentation
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.2)
- Normalization (ImageNet mean/std)

### Validation/Test
- No augmentation
- Center crop to 224x224
- Normalization (ImageNet mean/std)

## Storage Requirements

### Estimated Disk Space
- Raw data: ~50 MB (CSV from Hugging Face)
- Processed images: ~100-150 MB (4,500 images @ ~30 KB each)
- Checkpoints: ~10-50 MB (depending on model)
- Total: ~200-250 MB

### Caching
- Download cache in `data/cache/`
- Processed images in `data/processed/`
- Can be deleted and regenerated if needed

## Reproducibility

### Deterministic Pipeline
- Fixed random seed (42)
- Deterministic data splits
- Reproducible image processing
- Hash-based deduplication

### Re-running the Pipeline
```bash
# Delete processed data
rm -rf data/processed data/cache

# Re-download
python -m src.data.download
```

Should produce identical results (modulo download failures).

## Ethical Considerations

### Privacy
- No personal data in images
- Product images are public catalog photos

### Bias
- Dataset skewed toward Indian e-commerce
- May not generalize to other markets
- Brand bias toward Flipkart's inventory

### Fairness
- Balanced classes to avoid favoritism
- Stratified splits to maintain distribution

## Data Provenance

1. **Original Source**: Flipkart.com (scraped by PromptCloud)
2. **Kaggle**: Uploaded to Kaggle (original source unknown)
3. **Hugging Face**: Converted to Hugging Face dataset by jason1966
4. **This Project**: Downloaded, processed, and filtered

## Contact

For questions about the dataset, please refer to the original Hugging Face dataset page or open an issue in this repository.
