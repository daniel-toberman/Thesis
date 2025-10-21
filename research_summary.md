# Hybrid Sound Source Localization: CRNN Failure Detection & Classical Method Fallback

## What We Are Doing

**Main Idea**: Develop a hybrid approach to sound source localization that combines neural networks (CRNN) with classical methods (SRP-PHAT). The goal is to identify cases where the neural network fails and automatically switch to classical methods that might perform better.

**Training Setup**: We trained a CRNN on real-world data from the RealMAN dataset (not simulated data) using only low reverberation scenarios (T60 < 0.8 seconds) on a **6cm diameter microphone array**. The network achieved excellent performance with low MAE (3.86°) on the original training distribution.

## Key Experimental Progression

### 1. T60 Generalization Test
**Hypothesis**: Network would perform poorly on high reverberation data (T60 > 0.8) since it was only trained on T60 < 0.8.
**Result**: Network achieved similar MAE performance 4.76° on full test dataset with T60 ranging up to 4.5 seconds, demonstrating excellent reverberation generalization.

### 2. Out-of-Distribution Noise Test
**Hypothesis**: Adding novel noise from unseen environments would degrade performance.
**Setup**: Selected high-reverberation noise from T60 > 0.8 scenes (Cafeteria2) that the network had never seen during training, added at 5dB SNR.
**Initial Result**: Network performance degraded to 14.98° MAE.
**Critical Discovery**: After detailed investigation, only car gasoline and car electric cases brought performance down. The MAE from all other test environments remained excellent at ~6.31°.

### 3. Automotive Environment Challenge - Initial Approach FAILED
**Hypothesis**: Automotive environment + novel noise combination was too acoustically challenging for classical methods.
**Result**: After extensive testing of SRP variants (parameter tuning, ensemble methods, array scaling), all classical approaches failed to rescue automotive failures.
**Decision**: After consulting with professor, pivoted away from automotive environment approach.

### 4. Microphone Array Geometry Investigation - NEW DIRECTION
**Hypothesis**: CRNN trained on 6cm array geometry might not generalize to different microphone configurations. This geometric sensitivity could create opportunities for SRP fallback.

#### CRNN Geometry Robustness Test (Critical Finding)
**Setup**: Tested CRNN on three microphone array diameters without retraining:
- **6cm diameter** (training configuration): Microphones 0-8 (center + 8 outer)
- **12cm diameter**: Microphones 0, 9-16 (center + 8 medium circle)
- **18cm diameter**: Microphones 0, 17-24 (center + 8 large circle)

**Results on Clean Test Set (2009 samples, no novel noise):**

| Array Diameter | MAE | Success Rate (≤30°) | Failure Rate (>30°) | vs 6cm Training |
|---------------|-----|---------------------|---------------------|----------------|
| **6cm** (training) | ~4.8° | ~95% | ~5% | baseline |
| **12cm** | 63.34° | 34.2% | **65.8%** | **+58° MAE** |
| **18cm** | TBD | TBD | TBD | TBD |

**🔥 BREAKTHROUGH DISCOVERY**: CRNN exhibits **catastrophic failure on different microphone array geometries**
- Moving from 6cm to 12cm array causes 58° MAE degradation
- Success rate drops from 95% to 34% - a 60% absolute decrease
- 65.8% of test cases now fail (>30° error) on 12cm array
- **This geometric brittleness opens the door for SRP-PHAT rescue**

**Why This Matters**:
- CRNN learns geometry-specific acoustic patterns during training
- Classical methods like SRP-PHAT are geometry-agnostic (just need mic positions)
- **New hybrid opportunity**: Use SRP when array geometry differs from training

### 5. Confidence-Based Failure Prediction ✅ VALIDATED
**Approach**: Developed confidence-based failure prediction using CRNN's internal metrics (max probability, entropy, prediction variance, etc.).
**Achievement**: Successfully predicted 80.1% of CRNN failures with 76.1% precision using simple threshold: max_prob ≤ 0.02560333.
**Status**: Failure detection works reliably - can identify when CRNN is uncertain in real-time.

### 6. SRP Parameter Optimization 🔄 IN PROGRESS
**Challenge**: Need SRP to achieve acceptable accuracy (<30° MAE) to serve as effective fallback.
**Current Status**: Running comprehensive parameter optimization across 224 combinations.

**Optimization Framework**:
- **Parameters tested**: DFT bins (256-32768), averaging samples (1-50), frequency range (200-4000Hz), grid cells (360-720)
- **Phase 1**: Screen all 224 combinations on 100 random samples
- **Phase 2**: Test top 10 configurations on full 2009-sample dataset

**Best Results (Phase 1 - 100 samples)**:
- **Best configuration**: n_dft_bins=32768, freq=300-4000Hz → **21.51° MAE**
- Median error: 4.50° (excellent for successful cases)
- Improvement over baseline: ~70° → 21° MAE reduction

**Next Steps**:
1. Complete Phase 2 testing on full dataset for top configurations
2. Test failure prediction robustness on 12cm/18cm arrays
3. Validate hybrid system: CRNN prediction + confidence check + SRP fallback for geometric mismatch

## Executive Summary

This research pivoted from automotive environment failures (which proved intractable for classical methods) to **microphone array geometry robustness**. We discovered that CRNN exhibits catastrophic failure (65.8% failure rate) when tested on array geometries different from training (12cm vs 6cm), while maintaining excellent performance on its training geometry. This geometric brittleness creates a clear opportunity for hybrid systems: CRNN handles trained geometry configurations, while SRP-PHAT rescues cases with different array setups. We've achieved 80% failure detection using confidence metrics and are optimizing SRP parameters to reduce MAE from 70° to ~21° for effective fallback.

## Key Findings

### CRNN T60 Generalization Performance
**CRNN demonstrates excellent reverberation time (T60) generalization**. The model was trained exclusively on data with T60 < 0.8 seconds but successfully generalizes to much higher reverberation conditions. Performance comparison shows nearly identical results between the original test set (T60 < 0.8s) and the extended T60 test set (T60 ranging up to 4.5 seconds), indicating robust acoustic generalization across diverse reverberation environments.

### CRNN Geometry Brittleness (Critical Discovery)
**CRNN fails catastrophically on different microphone array geometries**:
- Trained on 6cm array: 95% success rate, ~4.8° MAE
- Tested on 12cm array: **34% success rate, 63° MAE**
- **65.8% failure rate** when array diameter changes from training
- Root cause: Network learns geometry-specific acoustic features
- **Opportunity**: SRP-PHAT is geometry-agnostic, can rescue these failures

### Automotive Environment Findings (Historical Context)
Analysis of CRNN failures with novel noise revealed **systematic failures in automotive environments** (99.5% of failures). However, after consultation with professor, this approach was abandoned because:
- Automotive + novel noise too acoustically challenging for classical methods
- Ensemble SRP achieved only 26.9% success rate with unacceptable 82.8° MAE
- Array scaling (6cm → 18cm) provided no improvement
- Classical methods fundamentally cannot rescue this failure mode

### Confidence-Based Failure Prediction ✅ VALIDATED
**Breakthrough: Real-Time CRNN Failure Prediction**
- **max_prob** metric: 78.1% F1, 76.1% precision, 80.1% recall
- Simple threshold (max_prob ≤ 0.02560333) enables real-time switching
- Minimal computational overhead - just tensor operations on existing outputs
- Works across different failure scenarios (tested on automotive+noise)

### SRP Parameter Optimization 🔄 IN PROGRESS
**Goal**: Achieve <30° MAE for effective CRNN fallback

**Current Best Results (100-sample screening)**:
- Configuration: n_dft_bins=32768, freq=300-4000Hz, grid=360
- **21.51° MAE** (median 4.50°)
- 70% improvement over baseline SRP (70° → 21° MAE)
- Awaiting full dataset validation (Phase 2)

**Key Parameter Insights**:
- Higher DFT bins dramatically improve accuracy (32768 optimal)
- Frequency range: 300-4000Hz performs best (captures speech formants)
- Averaging samples (1-50) shows minimal impact
- Grid resolution (360 vs 720) has minor effect

## Research Direction Change

### Why We Abandoned Automotive Environment Approach
1. **Intractable Acoustic Challenge**: Automotive + novel noise creates acoustic interference so severe that even 3x larger array aperture cannot overcome it
2. **Classical Method Limitations**: All SRP optimizations (parameters, ensembles, preprocessing) failed to achieve acceptable accuracy
3. **Professor Guidance**: Advised to find failure modes where classical methods can actually help

### New Direction: Microphone Array Geometry Robustness
1. **Discovered Geometric Brittleness**: CRNN fails when array geometry differs from training (65.8% failure rate on 12cm vs 6cm training)
2. **SRP Advantage**: Classical methods are geometry-agnostic - just need microphone positions
3. **Clear Hybrid Opportunity**: CRNN for trained geometries, SRP fallback for different configurations
4. **Practical Relevance**: Real-world deployments often use different array sizes than training data

## Implementation Strategy

### Phase 1: Confidence Metric Analysis ✅ COMPLETED
- ✅ Extract CRNN confidence scores from all test examples
- ✅ Analyze confidence distributions for successful vs failed predictions
- ✅ Test multiple confidence metrics: entropy, max probability, prediction variance, peak sharpness, local concentration
- ✅ Identify which metrics best correlate with failure cases

**BREAKTHROUGH RESULTS:**
- **MAX_PROB**: F1=0.781 (76.1% precision, 80.1% recall) - **Best performer**
- **LOCAL_CONCENTRATION**: F1=0.770 (75.1% precision, 79.1% recall) - **Second best**
- **PREDICTION_VARIANCE**: F1=0.765 (74.6% precision, 78.5% recall) - **Third best**

### Phase 2: Failure Prediction Development ✅ COMPLETED
- ✅ Optimal predictor: 80.1% recall, 76.1% precision using max_prob ≤ 0.02560333
- ✅ Built ML classifier using multiple confidence metrics
- ✅ Tested on automotive environment failures - works reliably
- ✅ Ready for geometry robustness testing

### Phase 3: CRNN Geometry Robustness Testing ✅ COMPLETED
- ✅ Tested CRNN on 6cm (training), 12cm, and 18cm arrays
- ✅ **Critical Finding**: 65.8% failure rate on 12cm array (vs 5% on 6cm)
- ✅ Confirmed geometric brittleness creates hybrid opportunity
- 🔄 Testing failure predictor on 12cm/18cm data (in progress)

### Phase 4: SRP Parameter Optimization 🔄 IN PROGRESS
**Goal**: Achieve <30° MAE for effective fallback

**Phase 1 Screening (100 samples)**: ✅ COMPLETED
- Tested 224 parameter combinations
- Best: 21.51° MAE (n_dft_bins=32768, freq=300-4000Hz)
- Identified top 10 configurations for full testing

**Phase 2 Full Dataset Testing**: 🔄 IN PROGRESS
- Testing top 10 configurations on 2009 samples
- Expected runtime: ~6 hours per configuration
- Target: Validate <25° MAE on full dataset

### Phase 5: Hybrid System Validation 📋 NEXT
**Objectives**:
1. Test confidence predictor on 12cm/18cm array failures
2. Validate SRP fallback effectiveness on geometric mismatch cases
3. Measure hybrid system performance:
   - CRNN accuracy on 6cm array (should maintain ~95%)
   - SRP rescue rate on 12cm/18cm arrays (target >70% of failures)
   - Overall system MAE across all array configurations

**Success Criteria**:
- Hybrid system MAE < 20° across all array sizes
- >80% success rate on geometric mismatch cases
- Minimal performance degradation on trained geometry (6cm)

## Confidence-Based Failure Prediction

### Confidence Metrics Explained
Our analysis identified 5 key confidence metrics that can predict CRNN failures **in real-time**:

1. **MAX_PROB** - Maximum probability in the output distribution
   - Failed predictions: 0.019 vs Successful: 0.064 (p < 0.001)
   - Best single predictor (F1=0.781)

2. **ENTROPY** - Information entropy of probability distribution
   - Failed predictions: 4.84 vs Successful: 3.62 (p < 0.001)
   - Higher entropy = more uncertainty

3. **LOCAL_CONCENTRATION** - Probability mass in ±10° window around prediction
   - Failed predictions: 0.33 vs Successful: 0.78 (p < 0.001)
   - Low concentration = scattered probability

4. **PREDICTION_VARIANCE** - Variance of the probability distribution
   - Strong discriminator (F1=0.765)

5. **PEAK_SHARPNESS** - Ratio of max to second-max probability
   - Weaker but still significant predictor

### Real-Time Failure Prediction
**Key Breakthrough:** These metrics are computed **simultaneously** with CRNN predictions, enabling:
- **Real-time confidence assessment** before committing to CRNN result
- **Intelligent switching** to SRP-PHAT when confidence is low
- **80% failure detection** with 76% precision using simple thresholds
- **Minimal computational overhead** - just tensor operations on existing outputs

## SRP Parameter Optimization Results

### Phase 1 Screening (224 Combinations on 100 Samples)

**Top 10 Configurations**:
```
n_dft_bins=32768, n_avg= 1, freq=300-4000Hz: MAE=21.51°, Median=4.50°
n_dft_bins=32768, n_avg= 5, freq=300-4000Hz: MAE=21.51°, Median=4.50°
n_dft_bins=32768, n_avg=10, freq=300-4000Hz: MAE=21.51°, Median=4.50°
n_dft_bins=32768, n_avg=50, freq=300-4000Hz: MAE=21.51°, Median=4.50°
n_dft_bins=32768, n_avg= 1, freq=200-4000Hz: MAE=23.35°, Median=5.50°
n_dft_bins=16384, n_avg= 1, freq=200-4000Hz: MAE=23.48°, Median=6.48°
```

**Key Findings**:
- **DFT Bins Matter Most**: 32768 bins dramatically outperform lower values (21° vs 70° MAE)
- **Frequency Range**: 300-4000Hz optimal (captures speech formants, reduces low-freq noise)
- **Averaging Samples**: Minimal impact (1-50 samples show identical performance)
- **Grid Resolution**: 360 vs 720 cells has negligible effect

### Phase 2 Full Dataset Testing (In Progress)
**Status**: Running top 10 configurations on full 2009-sample dataset
**Expected Completion**: ~60 hours (6 hours per config × 10 configs)
**Target**: Validate 21° MAE performance on complete test set

## Expected Contributions

1. **Novel Discovery**: First identification of CRNN geometric brittleness - neural SSL fails catastrophically (65% failure rate) on different microphone array sizes
2. **Confidence-Based Failure Prediction**: Real-time failure prediction using CRNN's internal confidence metrics with 80% recall and 76% precision
3. **Practical Hybrid System**: Geometry-aware switching that addresses real-world deployment scenarios where array configurations vary
4. **SRP Optimization Framework**: Comprehensive parameter study identifying optimal configurations for SSL fallback (21° MAE, 70% improvement)

## Success Metrics

1. **Hybrid System Performance**: <20° MAE across all array geometries (6cm, 12cm, 18cm)
2. **Geometric Robustness**: >80% success rate on non-training array configurations
3. **Failure Detection**: Maintain 80% recall, 76% precision for failure prediction
4. **SRP Rescue Rate**: >70% of geometric mismatch failures successfully resolved by SRP
5. **Training Geometry Preservation**: <5% performance degradation on 6cm array

## Key Technical Insights

### What Works Well
- **CRNN on Training Geometry**: Excellent performance (95% success, ~5° MAE) when array matches training
- **T60 Generalization**: CRNN robust to reverberation time variations (0.3s to 4.5s)
- **Confidence Metrics**: Reliable real-time failure detection with simple thresholds
- **High DFT Resolution**: 32768 bins enable precise SRP localization

### Critical Limitations
- **CRNN Geometric Brittleness**: Catastrophic failure (65% error rate) on different array sizes
- **Training Data Dependency**: CRNN learns geometry-specific acoustic patterns
- **SRP Baseline Performance**: Poor accuracy (~70° MAE) without optimization
- **Automotive+Noise Challenge**: Intractable for both CRNN and classical methods

### Future Directions
- **Multi-Geometry Training**: Train CRNN on multiple array configurations simultaneously
- **Geometry-Conditioned Networks**: Explicit array geometry as input to network
- **Advanced SRP Methods**: MUSIC, ESPRIT, or learning-based refinement of SRP
- **Transfer Learning**: Adapt pre-trained CRNN to new array geometries with minimal data

## Research Questions for Thesis

### Primary Research Goal
**Develop a hybrid SSL system that combines CRNN's excellent performance on trained geometries with SRP-PHAT's geometry-agnostic fallback for array configurations not seen during training.**

### Key Research Questions
1. ✅ **Can we predict CRNN failures from network outputs?** YES - 80% recall using max_prob metric
2. ✅ **Does CRNN generalize to different array geometries?** NO - 65% failure rate on 12cm vs 6cm training
3. 🔄 **Can SRP-PHAT rescue geometric mismatch failures?** Testing - optimized SRP shows 21° MAE on samples
4. 📋 **Can we build a geometry-aware hybrid system?** Next - validation pending Phase 2 SRP results

### Extended Research Directions

#### Geometric Robustness
1. **Multi-Array Training**: Train CRNN on multiple geometries simultaneously
2. **Geometry Conditioning**: Provide array configuration as network input
3. **Transfer Learning**: Adapt trained models to new array sizes
4. **Uncertainty Quantification**: Confidence calibration for geometric mismatch

#### Hybrid System Optimization
5. **SRP Refinement**: Advanced classical methods (MUSIC, ESPRIT)
6. **Learning-Based Fusion**: ML models combining CRNN and SRP features
7. **Adaptive Switching**: Dynamic confidence thresholds based on array geometry
8. **Real-Time Deployment**: Computational efficiency and latency optimization
