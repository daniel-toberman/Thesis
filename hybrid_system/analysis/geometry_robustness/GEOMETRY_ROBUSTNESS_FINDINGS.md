# CRNN Geometry Robustness Analysis - Key Findings

## Executive Summary

**Major Discovery**: CRNN shows severe vulnerability to microphone array geometry changes, revealing a fundamental limitation in its robustness that creates a genuine hybrid opportunity.

## Quantitative Results

### CRNN Performance Degradation
- **6cm Array (baseline)**: 3.8° MAE on clean test data
- **12cm Array**: 63.34° MAE on clean test data
- **Degradation**: 59.54° increase (1566% worse performance)

### CRNN 12cm Failure Analysis
- **Total test cases**: 2,009
- **Catastrophic failures (>90° error)**: 486 cases (24.2%)
- **Extreme failures (>150° error)**: 264 cases (13.1%)
- **Acceptable performance (≤30° error)**: 688 cases (34.2%)

### Systematic Bias Patterns
CRNN 12cm shows strong systematic biases, clustering predictions around:
- **348°**: 173 cases (8.6%)
- **0°**: 113 cases (5.6%)
- **320°**: 83 cases (4.1%)
- **123°**: 74 cases (3.7%)

### Confidence Degradation
- **max_prob**: 0.029 (vs ~0.05-0.07 for good predictions)
- **entropy**: 4.72 (vs ~3.5-3.8 for confident predictions)
- **prediction_variance**: 0.000034 (elevated uncertainty)

## Key Research Insights

### 1. CRNN Robustness Has Limits
- **Robust to**: Novel noise, T60 variations, SNR changes
- **Vulnerable to**: Microphone array geometry changes
- **Implication**: Neural SSL models may have blind spots in physical array configurations

### 2. Failure Patterns Are Systematic, Not Random
- Not uniform degradation across all angles
- Strong clustering around specific wrong directions (348°, 0°, 320°, 123°)
- Suggests learned spatial representations break down with geometry mismatch

### 3. Confidence Metrics Successfully Detect Geometry Failures
- Low max_prob (0.029 vs 0.05+ for good predictions)
- High entropy (4.72 vs 3.5 for confident predictions)
- **Hybrid opportunity**: Use confidence to switch to classical methods

## Research Implications

### 1. Genuine Hybrid Opportunity Identified
- **Hypothesis**: Classical SRP-PHAT may be more geometry-robust than CRNN
- **Test in progress**: SRP performance on 6cm vs 12cm arrays
- **Expected**: SRP degradation should be much smaller than CRNN's 1566% increase

### 2. Geometry-Aware SSL Systems
- Arrays in real applications may vary due to:
  - Manufacturing tolerances
  - Installation constraints
  - Thermal expansion
  - Vibration/mounting effects
- **Need**: Robust SSL that adapts to geometry variations

### 3. Neural SSL Deployment Considerations
- Current CRNN training assumes fixed 6cm geometry
- **Limitation**: Poor generalization to different array configurations
- **Solution**: Either geometry-adaptive training or hybrid switching

## Next Steps

### Immediate Analysis
1. **SRP Geometry Comparison**: Measure SRP performance on 6cm vs 12cm
2. **Quantitative Robustness**: Compare CRNN vs SRP geometry degradation
3. **Confidence-Based Switching**: Test prediction accuracy for geometry failure detection

### Research Extensions
1. **Gradient Analysis**: Why does CRNN fail at 12cm? Input sensitivity analysis
2. **Geometry Interpolation**: Test intermediate array sizes (8cm, 10cm)
3. **Multi-Array Training**: Train CRNN on multiple array geometries
4. **Real-World Validation**: Test on actual hardware with geometry variations

## Methodology Notes

### Test Configuration
- **Dataset**: Clean RealMAN T60=0.8s test set (2,009 cases)
- **6cm Array**: Microphones [0,1,2,3,4,5,6,7,8] (original training configuration)
- **12cm Array**: Microphones [0,9,10,11,12,13,14,15,16] (center + outer ring)
- **Model**: Best CRNN checkpoint (08_CRNN/checkpoints/best_valid_loss0.0220.ckpt)

### Confidence Metrics
- **max_prob**: Maximum softmax probability
- **entropy**: Prediction entropy (uncertainty measure)
- **prediction_variance**: Variance of softmax distribution
- **peak_sharpness**: Sharpness of probability peak
- **local_concentration**: Local confidence around prediction

---

**Generated**: $(date)
**Analysis**: CRNN Geometry Robustness Investigation
**Status**: SRP comparison in progress