# CRNN Failure Analysis - Hybrid Approach Opportunities

## Executive Summary
Analysis of 3,582 examples reveals CRNN achieves excellent overall performance (3.52° MAE) but has **specific, predictable failure modes** that present clear opportunities for hybrid classical-neural approaches.

## Key Findings

### Performance Overview
- **Overall Performance**: 3.52° MAE, 2.0° median
- **Excellent Cases**: 96.0% have <10° error
- **Failure Cases**: Only 0.98% have >20° error, but these can be severe (up to 101°)

### Comparative Analysis: CRNN vs SRP-PHAT
- **CRNN Excellence**: Rarely fails (0.98% >20° error)
- **SRP Excellence**: Rarely excels (2% <2° error on random samples)

### Validated Research Findings
- ✅ **T60 Generalization**: CRNN handles T60 up to 4.9 with only 0.11° degradation

### Systematic Failure Patterns

#### Pattern 1: 137° Prediction Bias
**Affected Ground Truth Range**: 100-120° region
- **Failure Count**: 16 cases
- **Typical Error**: CRNN predicts ~137° when truth is 107-116°
- **Error Magnitude**: 20-37° errors

#### Pattern 2: 1° Prediction Bias
**Affected Ground Truth Range**: 20-40° region
- **Failure Count**: 9 cases
- **Typical Error**: CRNN predicts ~1° when truth is 22-36°
- **Error Magnitude**: 20-35° errors

#### Pattern 3: Large Backward Bias
**Affected Ground Truth Range**: 160-195° region
- **Failure Count**: 9 cases
- **Typical Error**: CRNN underestimates by 24-86°
- **Worst Case**: GT: 195°, Pred: 109° (86° error)

### Opportunity Analysis

#### Target Population for Hybrid Approaches
1. **Primary Target**: 35 cases with >20° error (0.98% of dataset)
2. **Secondary Target**: 162 cases with >10° error (4.52% of dataset)
3. **Maximum Benefit**: Reduce worst-case from 101° to ~20° range

#### Hybrid Strategy Recommendations

1. **Failure Prediction Model**
   - Train classifier to detect when CRNN will fail (>20° error)
   - Features: Raw audio, CRNN confidence, predicted angle
   - Target: 35 positive examples out of 3,582

2. **Angular Region Switching**
   - **Region 1**: 20-40° - Switch to SRP when CRNN predicts ~1°
   - **Region 2**: 100-120° - Switch to SRP when CRNN predicts ~137°
   - **Region 3**: 160-195° - Verify CRNN with SRP for large backward predictions

3. **Confidence-Based Ensemble**
   - Use both CRNN and SRP when prediction falls in failure-prone regions
   - Weight combination based on historical performance per region

## Research Questions for Thesis

### Primary Questions
1. **Can we predict CRNN failures?** Build a model to identify the 0.98% failure cases
2. **Do classical methods help in failure cases?** Test SRP-PHAT on the 35 worst examples
3. **What causes systematic biases?** Analyze audio characteristics of 137° and 1° prediction cases

### Extended Research Directions

#### Out-of-Distribution Testing
4. **Novel Noise Types**: Test CRNN on different noise types (not SNR changes) while keeping T60 < 0.8
   - Current training: Uses specific noise from RealMAN dataset
   - **Research Question**: Does CRNN fail with unseen noise types where SRP remains robust?
   - **Hypothesis**: Classical methods may be more noise-agnostic than neural networks

5. **Reverberation Robustness - VALIDATED**: ✅ **COMPLETED ANALYSIS**
   - **Experiment**: Tested T60 < 0.8 trained model on full test set (T60 up to 4.9)
   - **Results**: 4.76° MAE vs 4.65° MAE (only 0.11° degradation)
   - **Conclusion**: CRNN shows excellent T60 generalization - **T60 is NOT a viable hybrid differentiator**
   - **High T60 test coverage**: 49.8% of full test set includes T60 ≥ 0.8
   - **Extreme cases**: Successfully handled T60 = 4.9 (UndergroundParking2)

6. **Cross-Domain Robustness**: Evaluate on different acoustic environments
   - Test trained model on entirely different room types/sizes
   - Assess if SRP's physics-based approach provides better generalization

### Technical Investigations
2. **Uncertainty Quantification**: Can CRNN confidence scores predict failures?
5. **Complementary Failure Modes**: **PRIMARY FOCUS** - Find noise-based scenarios where CRNN fails but SRP succeeds

## Implementation Strategy

### Phase 1: Failure Case Analysis
- Extract audio files for the 35 worst cases
- Run SRP-PHAT on these specific examples
- Compare classical vs neural performance on failure cases

### Phase 2: Predictive Model
- Build failure detection classifier
- Features: CRNN outputs, confidence, spectral characteristics
- Validate on held-out failure cases

### Phase 3: Hybrid System
- Implement region-based switching logic
- Test on full dataset with hybrid approach
- Measure improvement in worst-case performance

### Phase 4: Out-of-Distribution Robustness Testing
- **Novel Noise Experiment**: Create test sets with different noise types
  - Urban noise (traffic, construction) vs original dataset noise
  - Industrial noise (machinery, fans) vs original noise
  - Natural noise (wind, rain) vs original noise
  - **Key**: Same SNR levels, different noise characteristics
- **Cross-Environment Testing**: Test on recordings from different room acoustics
- **Degradation Analysis**: Compare CRNN vs SRP performance degradation curves

## Expected Contributions

1. **Systematic Analysis**: First detailed analysis of CRNN failure modes in SSL
2. **Targeted Hybrid Approach**: Novel method focusing on failure case mitigation
3. **Predictive Framework**: Model to identify when neural networks will fail
4. **Practical Impact**: Improve worst-case reliability without hurting overall performance

## Success Metrics

1. **Failure Reduction**: Reduce >20° error cases from 35 to <10
2. **Worst-Case Improvement**: Reduce maximum error from 101° to <30°
3. **Overall Preservation**: Maintain 3.52° overall MAE performance
4. **Computational Efficiency**: Minimize classical method usage to failure cases only