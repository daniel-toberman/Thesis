# ConfidNet: Learned Failure Detection for Hybrid CRNN-SRP System

This directory contains a complete implementation of ConfidNet-based failure detection for the hybrid CRNN-SRP Direction-of-Arrival (DOA) estimation system.

## Overview

**ConfidNet** (Corbière et al., NeurIPS 2019) learns to predict when the CRNN model will fail by training a secondary neural network on the penultimate layer features. Cases predicted to fail are routed to SRP-PHAT for more robust estimation.

## Architecture

```
Input: Penultimate Features (256-dim from CRNN)
  ↓
Dense Layer (256 → 128)
  ↓
BatchNorm + ReLU + Dropout (0.3)
  ↓
Dense Layer (128 → 64)
  ↓
BatchNorm + ReLU + Dropout (0.3)
  ↓
Dense Layer (64 → 1)
  ↓
Sigmoid → Confidence Score [0, 1]
```

## Files

- `confidnet_model.py` - Model architecture and loss function
- `train_confidnet.py` - Training script with train/val split
- `confidnet_routing.py` - Routing logic and threshold optimization
- `evaluate_confidnet_hybrid.py` - Full hybrid system evaluation with SRP

## Quick Start

### Step 1: Test Model Architecture

```bash
python confidnet_model.py
```

This will test the model architecture and verify everything is working.

### Step 2: Test Training (Quick Mode)

Before running the full training, test on a small subset:

```bash
python train_confidnet.py \
  --epochs 10 \
  --test_mode \
  --device mps
```

Expected output:
- Training on 1,000 samples
- Validation accuracy should improve quickly
- Creates `models/confidnet/best_model.ckpt`
- Generates training curves and confidence distribution plots

### Step 3: Full Training

Train on the complete training set (11,508 samples):

```bash
python train_confidnet.py \
  --epochs 100 \
  --batch_size 256 \
  --lr 0.001 \
  --error_threshold 5.0 \
  --val_split 0.15 \
  --device mps \
  --output_dir models/confidnet
```

**Training Time**: ~15-20 minutes on MPS (Apple Silicon)

**Parameters**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--error_threshold`: Threshold for "correct" label (default: 5.0°)
- `--val_split`: Validation split ratio (default: 0.15)
- `--device`: Device to train on (mps/cuda/cpu)
- `--hidden_dims`: Hidden layer sizes (default: [128, 64])
- `--dropout`: Dropout rate (default: 0.3)

**Outputs**:
- `models/confidnet/best_model.ckpt` - Best model checkpoint
- `models/confidnet/training_curves.png` - Training/validation curves
- `models/confidnet/confidence_distribution.png` - Confidence distributions
- `models/confidnet/training_metrics.npz` - Training metrics

**Expected Results**:
- Training accuracy: 95-98%
- Validation accuracy: 93-96%
- Model should learn to separate correct from incorrect predictions

### Step 4: Evaluate Hybrid System

Run the full hybrid evaluation with SRP on routed cases:

```bash
python evaluate_confidnet_hybrid.py \
  --model_path models/confidnet/best_model.ckpt \
  --device mps
```

This will:
1. Load the trained ConfidNet model
2. Find optimal confidence threshold on test set
3. Route low-confidence cases to SRP
4. Run SRP on all routed cases (takes ~12-24 hours)
5. Compute hybrid performance metrics
6. Save results and summary

**Outputs**:
- `results/confidnet_hybrid_srp_results.csv` - SRP predictions for routed cases
- `results/confidnet_hybrid_summary.csv` - Performance summary

**Alternative: Use Fixed Threshold**

If you already know the optimal threshold:

```bash
python evaluate_confidnet_hybrid.py \
  --model_path models/confidnet/best_model.ckpt \
  --confidence_threshold 0.5 \
  --device mps
```

## Expected Results

Based on similar architectures, we expect:

**Routing Metrics**:
- F1 Score: 0.75-0.85
- Precision: 0.70-0.80
- Recall: 0.75-0.85
- False Positive Rate: 10-15% (vs 20% for advanced method, 21.5% for simple threshold)
- Routing Rate: 30-45%

**Hybrid Performance** (vs CRNN-only baseline: 15.41° MAE, 8.16° median, 38.4% success):
- Expected MAE: 13-14° (1-2° improvement)
- Expected Median: 2.5-3.5° (5-6° improvement)
- Expected Success: 60-65% (20-25% improvement)
- Routing Accuracy: 75-85% (cases where SRP is actually better)

## Understanding the Results

### Training Curves

Check `models/confidnet/training_curves.png`:
- **Loss should decrease smoothly** - If it plateaus early, try increasing learning rate
- **Val loss should track train loss** - Large gap indicates overfitting (increase dropout)
- **Accuracy should reach 93%+** - Lower accuracy suggests model capacity issues

### Confidence Distribution

Check `models/confidnet/confidence_distribution.png`:
- **Correct predictions** should have high confidence (0.7-1.0)
- **Incorrect predictions** should have low confidence (0.0-0.5)
- **Clear separation** between the two distributions indicates good calibration

### Routing Decisions

From `evaluate_confidnet_hybrid.py` output:
- **Precision**: Of cases routed to SRP, how many actually needed it?
- **Recall**: Of cases that needed SRP, how many did we route?
- **F1 Score**: Harmonic mean (balance between precision and recall)
- **False Positive Rate**: How many good CRNN predictions did we unnecessarily route?
- **Routing Accuracy**: Of routed cases, in how many is SRP actually better?

### Hybrid Performance

- **MAE/Median**: Should improve over CRNN-only baseline
- **Success Rate**: Should increase significantly (targeting 60%+)
- **Catastrophic Rescue**: Should catch most failures >30°

## Troubleshooting

### Training Issues

**Problem**: Validation accuracy not improving
- **Solution**: Reduce learning rate to 0.0005 or 0.0001
- **Solution**: Increase model capacity (hidden_dims=[256, 128, 64])
- **Solution**: Check training data quality (run with --test_mode first)

**Problem**: Training loss decreases but validation loss increases
- **Solution**: Increase dropout to 0.4 or 0.5
- **Solution**: Add more regularization (increase weight_decay to 1e-3)
- **Solution**: Reduce model capacity

**Problem**: Model predicts all 0s or all 1s
- **Solution**: Check class balance in training data
- **Solution**: Use class weights in loss function
- **Solution**: Adjust error_threshold parameter

### Routing Issues

**Problem**: Too many cases routed to SRP (>60%)
- **Solution**: Increase confidence threshold
- **Solution**: Retrain with different error_threshold (try 10° instead of 5°)

**Problem**: Too few cases routed (<20%)
- **Solution**: Decrease confidence threshold
- **Solution**: Check if model is overconfident (retrain with lower dropout)

**Problem**: Low routing accuracy (<60%)
- **Solution**: Model may not be learning useful features
- **Solution**: Try different architecture (more layers or different sizes)
- **Solution**: Check if penultimate features are informative (visualize with t-SNE)

## Advanced Usage

### Custom Error Threshold

Train with different error threshold for "correct" classification:

```bash
# More strict (only ≤3° is "correct")
python train_confidnet.py --error_threshold 3.0

# More lenient (≤10° is "correct")
python train_confidnet.py --error_threshold 10.0
```

### Custom Architecture

```bash
# Deeper network
python train_confidnet.py --hidden_dims 256 128 64 32

# Wider network
python train_confidnet.py --hidden_dims 512 256

# Less regularization
python train_confidnet.py --dropout 0.1
```

### Cross-Validation

To avoid overfitting to validation set, you can manually split data:

1. Split training data into K folds
2. Train K models, each on different train/val split
3. Ensemble predictions or select best model

## Comparison with Other Methods

| Method | MAE | Median | Success | Routing | FP Rate | Complexity |
|--------|-----|--------|---------|---------|---------|------------|
| **CRNN-only** | 15.41° | 8.16° | 38.4% | 0% | - | Low |
| **Simple Threshold** | 14.56° | 3.65° | 57.0% | 39.1% | 21.5% | Very Low |
| **Advanced (Temp+Mahal)** | 14.26° | 3.06° | 63.3% | 50.0% | 20.0% | Medium |
| **ConfidNet** | **~13-14°** | **~2.5-3.5°** | **~60-65%** | **30-45%** | **~10-15%** | High |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{corbiere2019confidnet,
  title={Addressing Failure Prediction by Learning Model Confidence},
  author={Corbi{\`e}re, Charles and Thome, Nicolas and Bar-Hen, Avner and Cord, Matthieu and P{\'e}rez, Patrick},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## Next Steps

After evaluating ConfidNet:

1. **Compare with baselines** - Add results to research_summary.md
2. **Analyze failure cases** - Which types of errors does ConfidNet catch?
3. **Try ensemble** - Combine ConfidNet with Mahalanobis distance
4. **Experiment with architectures** - Try different network sizes/depths
5. **Test on other arrays** - Evaluate on 12cm, 18cm configurations

## Support

For issues or questions:
1. Check this README first
2. Review training curves and confidence distributions
3. Try test_mode first before full training
4. Verify feature extraction used correct mic order [1,2,3,4,5,6,7,8,0]
