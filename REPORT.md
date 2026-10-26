# Product Image Classifier - Final Report

## Executive Summary

This report presents the results of training a CNN classifier for e-commerce product image categorization. The model classifies product images into three categories: **Apparel**, **Electronics**, and **Home**.

**Key Results:**
- **Validation Accuracy**: TBD% (target: >85%)
- **Test Accuracy**: TBD%
- **Test F1 (macro)**: TBD
- **Model**: Custom CNN with 2.5M parameters

## Model Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| Validation Accuracy | TBD% |
| Test Accuracy | TBD% |
| Test Precision (macro) | TBD |
| Test Recall (macro) | TBD |
| Test F1 (macro) | TBD |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Apparel | TBD | TBD | TBD | TBD |
| Electronics | TBD | TBD | TBD | TBD |
| Home | TBD | TBD | TBD | TBD |

*Note: Actual metrics will be populated after running the evaluation pipeline.*

## Training Details

### Hyperparameters
- **Model**: Custom CNN (4 conv blocks, 2 FC layers)
- **Optimizer**: Adam (lr=0.001, weight_decay=0.0001)
- **Batch Size**: 32
- **Epochs**: TBD (early stopping after 10 epochs without improvement)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Image Size**: 224x224
- **Augmentation**: Random flip, rotation, color jitter

### Training Progress
- **Best Epoch**: TBD
- **Training Time**: TBD
- **Final Training Loss**: TBD
- **Final Validation Loss**: TBD

Training curves are available in `outputs/figures/training_curves.png`.

## Confusion Matrix Analysis

### Confusion Matrix

The confusion matrix (normalized) shows:

```
              Predicted
              Apparel  Electronics  Home
True  Apparel    TBD%       TBD%     TBD%
      Electronics TBD%       TBD%     TBD%
      Home       TBD%       TBD%     TBD%
```

Full confusion matrix visualization: `outputs/figures/confusion_matrix.png`

### Top Confused Category Pairs

The following pairs show the highest confusion rates:

1. **[Class A] → [Class B]**: TBD samples (TBD%)
   - **Likely Cause**: TBD
   - **Evidence**: TBD
   - **Examples**: See `outputs/figures/misclassified_[A]_[B].png`

2. **[Class C] → [Class D]**: TBD samples (TBD%)
   - **Likely Cause**: TBD
   - **Evidence**: TBD
   - **Examples**: See `outputs/figures/misclassified_[C]_[D].png`

3. **[Class E] → [Class F]**: TBD samples (TBD%)
   - **Likely Cause**: TBD
   - **Evidence**: TBD
   - **Examples**: See `outputs/figures/misclassified_[E]_[F].png`

### Analysis of Confusion Patterns

#### Pattern 1: Visual Similarity
Some confusion arises from visual similarity between categories. For example:
- **Apparel accessories** (bags, scarves) may be confused with **Home decor** items
- **Small electronics** (headphones, smartwatches) may resemble **accessories**

#### Pattern 2: Background Context
The model may rely on background cues:
- **Home products** often have domestic backgrounds
- **Electronics** may be shown on white/grey studio backgrounds
- **Apparel** may include mannequins or models

#### Pattern 3: Label Noise
Original dataset labels may be inconsistent:
- Multi-category products (e.g., "smart home device")
- Unclear boundaries (e.g., "home appliances" vs "electronics")

#### Pattern 4: Class Imbalance
Despite balancing efforts, some subcategories may be underrepresented:
- Rare subcategories within a class may not be learned well
- Imbalanced visual features within a class

## Misclassification Examples

For each top confused pair, we provide a grid of misclassified examples:

### [Pair 1]
![Misclassified examples](outputs/figures/misclassified_[pair1].png)

**Analysis**: TBD

### [Pair 2]
![Misclassified examples](outputs/figures/misclassified_[pair2].png)

**Analysis**: TBD

### [Pair 3]
![Misclassified examples](outputs/figures/misclassified_[pair3].png)

**Analysis**: TBD

## Decision Output for Catalogue Team

### Confidence Threshold Analysis

Based on validation set analysis, we recommend:

**Auto-Accept Threshold**: TBD (e.g., 0.85)

- Predictions with confidence ≥ TBD are auto-accepted
- Predictions with confidence < TBD are routed to manual review
- This threshold achieves ~TBD% recall with ~TBD% manual review rate

### Routing Recommendations

1. **High-Confidence Predictions (≥ TBD%)**
   - Action: Auto-accept
   - Expected accuracy: ~TBD%
   - Volume: ~TBD% of predictions

2. **Medium-Confidence Predictions (TBD% - TBD%)**
   - Action: Manual review
   - Priority: Normal
   - Volume: ~TBD% of predictions

3. **Low-Confidence Predictions (< TBD%)**
   - Action: Manual review
   - Priority: High
   - Volume: ~TBD% of predictions

### Confusion-Based Routing

For specific confused pairs, consider:

- **[Class A] vs [Class B]**: These pairs are frequently confused. When the model predicts either with medium confidence, route to specialized reviewers familiar with both categories.

- **[Class C] vs [Class D]**: High confusion rate. Consider merging these categories or adding a "requires verification" flag.

### Operational Recommendations

1. **Active Learning**: Use manual review corrections to retrain the model periodically
2. **Data Quality**: Improve image quality standardization (background removal, consistent lighting)
3. **Category Refinement**: Consider splitting frequently confused subcategories
4. **Ensemble Methods**: Combine with metadata (text description, price) for better accuracy

## Limitations

1. **Dataset License**: Unknown license from Hugging Face; verify before production use
2. **Generalization**: Trained on Indian e-commerce data; may not generalize to other markets
3. **Image Quality**: Variable quality in downloaded images affects performance
4. **Background Bias**: Model may rely on background context rather than product features
5. **Single-Image Classification**: Does not use product metadata (text, price, etc.)

## Future Improvements

1. **Transfer Learning**: If baseline <85%, try MobileNetV2 or ResNet18 pretrained on ImageNet
2. **Multi-Modal**: Incorporate product text descriptions (title, category tree)
3. **Attention Mechanisms**: Add visual attention to focus on product rather than background
4. **Data Augmentation**: Add more aggressive augmentation (CutMix, MixUp)
5. **Ensemble**: Combine predictions from multiple models

## Conclusion

The product image classifier achieves TBD% test accuracy on a balanced dataset of ~4,500 images across 3 categories. The confusion matrix analysis reveals specific category pairs that require manual review, and we provide a confidence threshold recommendation for operational deployment.

**Recommendation**: Deploy with auto-accept threshold of TBD and route low-confidence predictions to manual review. Implement active learning to continuously improve the model with human feedback.

## Appendix

### A. Experiment Log

| Experiment | Model | Val Acc | Test Acc | Notes |
|------------|-------|---------|----------|-------|
| 1 | Custom CNN (baseline) | TBD% | TBD% | Initial run |
| 2 | Custom CNN + more augmentation | TBD% | TBD% | TBD |
| 3 | MobileNetV2 (transfer learning) | TBD% | TBD% | TBD |

### B. Files Generated

- `outputs/checkpoints/best_checkpoint.pth` - Best model checkpoint
- `outputs/figures/training_curves.png` - Loss and accuracy curves
- `outputs/figures/confusion_matrix.png` - Confusion matrix heatmap
- `outputs/figures/misclassified_*.png` - Misclassification example grids
- `outputs/metrics/evaluation_results.json` - Detailed metrics
- `outputs/metrics/training_history.json` - Training history

### C. Reproduction

To reproduce these results:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python -m src.data.download

# 3. Train model
python -m src.train.trainer

# 4. Evaluate
python -m src.evaluate.evaluator

# 5. View results
# Check outputs/ directory for figures and metrics
```

Or use the Colab notebook: `notebooks/colab_run.ipynb`

---

**Report Generated**: 2026-10-10
**Model Version**: TBD
**Dataset Version**: TBD
