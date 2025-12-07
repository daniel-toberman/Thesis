# OOD-Based Failure Detection: Evaluation Summary

## Overview

This document summarizes the implementation and evaluation of 4 OOD (Out-of-Distribution) detection methods for failure prediction in the hybrid CRNN-SRP system.

## Methods Implemented

### 1. Energy-Based OOD Detection ✅
**Status**: SUCCESS

**Approach**: Uses energy scores from CRNN logits: `E(x) = -T * log(Σ exp(logit_i / T))`
- Lower energy = confident prediction
- Higher energy = OOD/uncertain → route to SRP
- Temperature T calibrated on training data

**Implementation**:
- `train_energy_ood.py` - Temperature calibration
- `energy_ood_routing.py` - Router class
- No neural network training required

**Results**:
- Best F1 = 0.813 at 75% routing
- At 30% routing: F1 = 0.552, Precision = 0.842, Recall = 0.411
- Catastrophic capture rate: 80.8% at 30% routing

### 2. MC Dropout Ensemble (Entropy) ✅
**Status**: SUCCESS

**Approach**: Measures predictive uncertainty using entropy of CRNN logits
- Computes entropy: `-Σ p(i) log(p(i))` from softmax probabilities
- High entropy = high uncertainty → route to SRP
- Uses existing CRNN logits (no retraining needed)

**Implementation**:
- `mc_dropout_routing.py` - Router with entropy computation

**Results**:
- Best F1 = 0.792 at 80% routing
- At 30% routing: F1 = 0.557, Precision = 0.849, Recall = 0.414
- Catastrophic capture rate: 81.5% at 30% routing
- **Very similar performance to Energy OOD**

### 3. MC Dropout Ensemble (Variance) ❌
**Status**: POOR PERFORMANCE

**Approach**: Measures variance of CRNN logit distribution as uncertainty
- High variance should indicate uncertainty
- Uses existing CRNN logits

**Results**:
- Best F1 = 0.694 at 90% routing (not useful)
- At 30% routing: F1 = 0.313, Precision = 0.478, Recall = 0.233
- **Variance is not discriminative enough for this task**

### 4. Deep SVDD (One-Class Learning) ❌
**Status**: FAILED - HYPERSPHERE COLLAPSE

**Approach**: Learn hypersphere in feature space that encloses successful predictions
- Train network to minimize distance to hypersphere center
- Failures should fall outside hypersphere
- Trained only on successful predictions (addresses class imbalance)

**Implementation**:
- `train_deep_svdd.py` - Network training with hypersphere loss
- `deep_svdd_routing.py` - Router class

**Results**:
- **Complete failure**: All samples mapped to same point (0% routing)
- Known issue: "hypersphere collapse" documented in Deep SVDD literature
- Network learns degenerate solution despite regularization

**Why it failed**:
- Hypersphere collapse is a known failure mode of Deep SVDD
- Occurs when optimization finds trivial solution (all points at center)
- Would require more sophisticated training (e.g., soft-boundary Deep SVDD)

## Comparison with Baselines

### Performance at 30% Routing Rate:

| Method | F1 Score | Precision | Recall | Status |
|--------|----------|-----------|--------|--------|
| **ConfidNet (optimized)** | **0.640** | **0.888** | **0.500** | ✅ Best |
| MC Dropout (Entropy) | 0.557 | 0.849 | 0.414 | ✅ Viable |
| Energy OOD | 0.552 | 0.842 | 0.411 | ✅ Viable |
| ConfidNet (20° thresh) | ~0.45 | ~0.75 | ~0.35 | ✅ Baseline |
| MC Dropout (Variance) | 0.313 | 0.478 | 0.233 | ❌ Poor |
| Deep SVDD | 0.000 | 0.000 | 0.000 | ❌ Failed |

### Oracle Baselines:
- Oracle 25% routing: ~37.17° MAE
- Oracle 30% routing: ~33.34° MAE

## Key Findings

### ✅ Successful Approaches:
1. **ConfidNet remains best**: F1 = 0.640 at 35% routing with optimized threshold
2. **Energy OOD works**: State-of-the-art method shows promise, F1 = 0.552 at 30%
3. **MC Dropout Entropy works**: Similar to Energy OOD, F1 = 0.557 at 30%

### ❌ Failed Approaches:
1. **Deep SVDD collapsed**: Hypersphere collapse is a known failure mode
2. **MC Dropout Variance poor**: Not discriminative enough for this task

### 📊 Performance Insights:
- Energy OOD and MC Dropout Entropy perform nearly identically (F1 ≈ 0.55 at 30%)
- Both significantly lag behind ConfidNet (F1 = 0.64 at 35%)
- All methods show high precision (0.84-0.89) but struggle with recall
- At practical routing rates (25-35%), ConfidNet is clearly superior

### 💡 Why OOD Methods Underperform:

1. **Not trained for task**: OOD methods use generic uncertainty without task-specific calibration
2. **ConfidNet has supervision**: Explicitly trained to predict correctness
3. **Better at extremes**: OOD methods work better at higher routing rates (60-80%)
4. **Feature quality matters**: CRNN features may not capture right uncertainty signals

## Recommendations

### For Production Use:
✅ **Use ConfidNet with optimized threshold (0.45-0.50)**
- Best F1 score in practical routing range (20-35%)
- Well-calibrated with training data
- Explicitly trained for failure detection

### For Research/Exploration:
- Energy OOD: Simplest OOD method, no training required
- MC Dropout Entropy: Uses existing logits, computationally cheap
- Both viable for quick prototyping or when training ConfidNet is not feasible

### Not Recommended:
❌ Deep SVDD: Requires careful tuning to avoid collapse, not worth the effort
❌ MC Dropout Variance: Poor discrimination, not useful for this task

## Files Created

### Training Scripts:
- `train_energy_ood.py` - Energy OOD temperature calibration
- `train_deep_svdd.py` - Deep SVDD network training

### Router Classes:
- `energy_ood_routing.py` - Energy-based router
- `deep_svdd_routing.py` - Deep SVDD router
- `mc_dropout_routing.py` - MC Dropout router (both variants)

### Evaluation:
- `evaluate_ood_methods.py` - Universal OOD evaluation framework
- `run_all_ood_methods.py` - Master runner for all methods

### Results:
- `results/ood_methods/energy_ood/threshold_optimization.csv`
- `results/ood_methods/deep_svdd/threshold_optimization.csv`
- `results/ood_methods/mc_dropout_entropy/threshold_optimization.csv`
- `results/ood_methods/mc_dropout_variance/threshold_optimization.csv`
- `results/ood_methods/method_comparison.csv`

### Documentation:
- `OOD_METHODS_README.md` - Implementation guide
- `OOD_EVALUATION_SUMMARY.md` - This document

## Trained Models

### Successful Models:
- `models/energy_ood_20.0deg/energy_ood_model.pkl` (287 bytes)
- `models/confidnet_20deg/` - ConfidNet baseline
- `models/confidnet_30deg/` - ConfidNet 30° threshold

### Failed Models:
- `models/deep_svdd_20.0deg/deep_svdd_model.pkl` (176 KB, collapsed)

## Conclusion

**ConfidNet with threshold optimization remains the best approach for failure detection in the 20-35% routing range (F1 = 0.640).**

Energy OOD and MC Dropout Entropy are viable alternatives (F1 ≈ 0.55) but lag behind ConfidNet's task-specific training. Deep SVDD failed due to hypersphere collapse, and MC Dropout Variance shows poor discrimination.

The exploration of OOD methods validates that supervised confidence prediction (ConfidNet) outperforms generic uncertainty quantification for this specific task.
