# Hybrid Sound Source Localization: CRNN Failure Detection & Classical Method Fallback

## What We Are Doing

**Main Idea**: Develop a hybrid approach to sound source localization that combines neural networks (CRNN) with classical methods (SRP-PHAT). The goal is to identify cases where the neural network fails and automatically switch to classical methods that might perform better.

**Training Setup**: We trained a CRNN on real-world data from the RealMAN dataset (not simulated data) using only low reverberation scenarios (T60 < 0.8 seconds) on a **6cm diameter microphone array**. The network achieved excellent performance with low MAE (3.86°) on the original training distribution.

## Key Experimental Progression

### 1. T60 Generalization Test
**Hypothesis**: Network would perform poorly on high reverberation data (T60 > 0.8) since it was only trained on T60 < 0.8.
**Result**: Network achieved 4.76° MAE (median ~3.2°) on full test dataset with T60 ranging up to 4.5 seconds, demonstrating excellent reverberation generalization.

### 2. Out-of-Distribution Noise Test
**Hypothesis**: Adding novel noise from unseen environments would degrade performance.
**Setup**: Selected high-reverberation noise from T60 > 0.8 scenes (Cafeteria2) that the network had never seen during training, added at 5dB SNR.
**Initial Result**: Network performance degraded to 14.98° MAE (median ~8°).
**Critical Discovery**: After detailed investigation, only car gasoline and car electric cases brought performance down. The MAE from all other test environments remained excellent at ~6.31° (median ~4°).

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

| Array Diameter | MAE / Median | Success Rate (≤5°) | Failure Rate (>5°) | vs 6cm Training |
|---------------|--------------|-------------------|-------------------|----------------|
| **6cm** (training) | 2.82° / 2.0° | 87.8% | 12.2% | baseline |
| **12cm** (full) | 68.01° / 52.0° | **4.5%** | **95.5%** | **+65° MAE** |
| **18cm** (full) | 56.70° / 48.0° | **5.0%** | **95.0%** | **+54° MAE** |

**🔥 BREAKTHROUGH DISCOVERY**: CRNN exhibits **catastrophic failure on different microphone array geometries**
- Moving from 6cm to 12cm causes 65° MAE degradation (median: 50° increase), success drops from 87.8% to 4.5%
- Moving from 6cm to 18cm causes 54° MAE degradation (median: 46° increase), success drops from 87.8% to 5.0%
- **Interesting pattern**: 18cm performs moderately better than 12cm (56.7° vs 68.0° MAE), suggesting non-linear geometry degradation
- ~95% of test cases fail (>5° error) on non-training array geometries
- **This geometric brittleness opens the door for SRP-PHAT rescue**

#### Partial Microphone Replacement Test (NEW - Critical Finding)
**Hypothesis**: Testing gradual degradation by replacing only 1-2 microphones rather than entire array.

**Setup**: Starting with training configuration (6cm array), progressively replace 1-2 microphones with 12cm or 18cm positions:
- **Baseline**: All 8 outer mics at 6cm (mics 1-8, center mic 0 always at origin)
- **Single replacements**: Replace 1 mic at different positions (pos1, pos3, pos5)
- **Double replacements**: Replace 2 mics (opposite positions or adjacent)
- **Full replacement**: All 8 outer mics at 12cm or 18cm

**Results - Gradual Degradation Pattern (2009 samples):**

| Configuration | N_Replaced | Type | MAE / Median | Success (≤5°) | Degradation from Baseline |
|--------------|-----------|------|--------------|---------------|---------------------------|
| **6cm baseline** | 0 | none | 2.82° / 2.0° | 87.8% | baseline |
| **1x 18cm (pos1)** | 1 | 18cm | 3.34° / 2.0° | 82.2% | **+0.5° / -5.6%** |
| **1x 18cm (pos5)** | 1 | 18cm | 4.94° / 3.0° | 76.1% | **+2.1° / -11.7%** |
| **1x 12cm (pos5)** | 1 | 12cm | 5.92° / 3.0° | 68.1% | **+3.1° / -19.7%** |
| **1x 12cm (pos1)** | 1 | 12cm | 6.14° / 3.0° | 68.6% | **+3.3° / -19.2%** |
| **1x 12cm (pos3)** | 1 | 12cm | 6.19° / 4.0° | 58.4% | **+3.4° / -29.4%** |
| **2x 18cm (opp)** | 2 | 18cm | 7.08° / 3.0° | 66.1% | **+4.3° / -21.7%** |
| **2x 12cm (adj)** | 2 | 12cm | 10.06° / 4.0° | 57.0% | **+7.2° / -30.8%** |
| **2x 12cm (opp)** | 2 | 12cm | 15.58° / 6.0° | 45.8% | **+12.8° / -42.0%** |
| **12cm full** | 8 | 12cm | 68.01° / 52.0° | 4.5% | **+65.2° / -83.3%** |
| **18cm full** | 8 | 18cm | 56.70° / 48.0° | 5.0% | **+53.9° / -82.8%** |

**🎯 KEY INSIGHTS**:
1. **18cm is MUCH more robust than 12cm**: Single 18cm replacement (pos1) causes minimal degradation (3.34° MAE, 82% success) vs single 12cm (5.92-6.19° MAE, 58-69% success)
2. **Progressive degradation**: 0 → 1 → 2 replacements shows gradual performance loss, not catastrophic failure
3. **Position matters significantly**: pos1 better than pos5 for 18cm (3.34° vs 4.94°), pos5 better than pos3 for 12cm
4. **Opposite placement worse than adjacent**: 2x12cm opposite (15.58° MAE) much worse than adjacent (10.06° MAE)
5. **Cliff edge at full replacement**: Jump from 15.58° (2 mics) to 68.01° (8 mics) for 12cm - suggests network relies on majority of original geometry
6. **Hybrid opportunity refined**: Network remains usable with 1-2 mic replacements, especially with 18cm positions

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
- **Best configuration**: n_dft_bins=32768, freq=300-4000Hz → **21.51° MAE / 4.50° median**
- Median error of 4.50° indicates excellent performance on majority of cases
- Improvement over baseline: ~70° → 21° MAE reduction (74% improvement)

**Next Steps**:
1. Complete Phase 2 testing on full dataset for top configurations
2. Test failure prediction robustness on 12cm/18cm arrays
3. Validate hybrid system: CRNN prediction + confidence check + SRP fallback for geometric mismatch

## Executive Summary

This research pivoted from automotive environment failures (which proved intractable for classical methods) to **microphone array geometry robustness**. We discovered that CRNN exhibits catastrophic failure when tested on array geometries different from training: 12cm full array shows 68.0° MAE / 52° median (95.5% failure rate), while 18cm full array shows 56.7° MAE / 48° median (95.0% failure rate), compared to 2.82° MAE / 2° median on the 6cm training geometry.

**Critical new finding**: Partial microphone replacement tests reveal **gradual degradation** rather than binary failure. Single 18cm replacements cause minimal impact (3.34° MAE, 82% success), while single 12cm replacements show moderate degradation (5.92-6.19° MAE, 58-68% success). The non-linear pattern continues: 18cm positions are significantly more robust than 12cm despite being farther from training geometry. Position placement matters significantly, and there's a "cliff edge" - performance jumps from 15.58° (2 mics) to 68° (8 mics) for 12cm, suggesting the network relies on maintaining majority original geometry.

This geometric brittleness creates a clear opportunity for hybrid systems: CRNN handles trained geometry configurations and partial replacements (1-2 mics), while SRP-PHAT rescues cases with severe geometric mismatch. We've achieved 80% failure detection using confidence metrics and are optimizing SRP parameters to achieve 21.5° MAE / 4.5° median for effective fallback.

## Key Findings

### CRNN T60 Generalization Performance
**CRNN demonstrates excellent reverberation time (T60) generalization**. The model was trained exclusively on data with T60 < 0.8 seconds but successfully generalizes to much higher reverberation conditions. Performance comparison shows nearly identical results between the original test set (T60 < 0.8s) and the extended T60 test set (T60 ranging up to 4.5 seconds), indicating robust acoustic generalization across diverse reverberation environments.

### CRNN Geometry Brittleness (Critical Discovery)
**CRNN fails catastrophically on different microphone array geometries**:
- Trained on 6cm array: **87.8% success rate (≤5°), 2.82° MAE / 2.0° median**
- Tested on 12cm array (full): **4.5% success rate, 68.01° MAE / 52.0° median** (95.5% failure rate)
- Tested on 18cm array (full): **5.0% success rate, 56.70° MAE / 48.0° median** (95.0% failure rate)
- **Non-linear degradation**: 18cm performs better than 12cm despite being farther from training geometry
- **Gradual degradation discovered**: Partial replacements (1-2 mics) show progressive performance loss
  - Single 18cm replacement: 3.34-4.94° MAE (82-76% success) - **network remains highly usable**
  - Single 12cm replacement: 5.92-6.19° MAE (68-58% success) - **moderate degradation**
  - Two 12cm replacements: 10.06-15.58° MAE (57-46% success) - **severe degradation**
  - Full array replacement: **catastrophic failure** (~95% failure rate)
- Root cause: Network learns geometry-specific acoustic patterns that don't transfer
- **Opportunity**: SRP-PHAT is geometry-agnostic, can rescue these failures

### Automotive Environment Findings (Historical Context)
Analysis of CRNN failures with novel noise revealed **systematic failures in automotive environments** (99.5% of failures). However, after consultation with professor, this approach was abandoned because:
- Automotive + novel noise too acoustically challenging for classical methods
- Ensemble SRP achieved only 26.9% success rate (≤30°) with unacceptable 82.8° MAE / 75° median
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
- **21.51° MAE / 4.50° median**
- 70% improvement over baseline SRP (from 70° / 60° to 21° / 4.5°)
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
1. **Discovered Geometric Brittleness**: CRNN fails catastrophically when full array geometry differs from training (~95% failure rate at ≤5° on 12cm/18cm full arrays vs 6cm training)
2. **Gradual Degradation with Partial Replacement**: Progressive performance loss with 1-2 mic replacements (3-16° MAE) before cliff edge to catastrophic failure (68° MAE) with full array change
3. **Non-linear Degradation Pattern**: 18cm more robust than 12cm at all replacement levels (single: 3.34° vs 5.92°, double: 7.08° vs 10-15°, full: 56.7° vs 68°), suggesting complex acoustic feature dependencies
4. **Position-Dependent Sensitivity**: Specific microphone placements (pos1 vs pos3 vs pos5) significantly impact degradation severity
5. **SRP Advantage**: Classical methods are geometry-agnostic - just need microphone positions
6. **Clear Hybrid Opportunity**: CRNN for trained geometries and minor variations (1-2 mics), SRP fallback for severe geometric mismatch
7. **Practical Relevance**: Real-world deployments often use different array sizes than training data, and understanding partial degradation guides deployment decisions

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
- ✅ Tested CRNN on 6cm (training), 12cm, and 18cm full arrays
- ✅ **Critical Finding**: ~95% failure rate (>5° error) on full 12cm/18cm arrays (vs 12.2% on 6cm)
- ✅ **Non-linear Degradation**: 12cm shows 68.0° MAE / 52° median, 18cm shows 56.7° MAE / 48° median
- ✅ 18cm performs better than 12cm despite being farther from training geometry
- ✅ **NEW: Partial replacement tests** reveal gradual degradation (1-2 mics):
  - Single 18cm replacement: 3.34-4.94° MAE (82-76% success) - **minimal impact**
  - Single 12cm replacement: 5.92-6.19° MAE (58-68% success) - **moderate degradation**
  - Two 12cm replacements: 10.06-15.58° MAE (46-57% success) - **severe degradation**
  - "Cliff edge" phenomenon: 15.58° (2 mics) → 68° (8 mics) for 12cm
- ✅ Confirmed geometric brittleness creates hybrid opportunity
- 🔄 Testing failure predictor on 12cm/18cm data (in progress)

### Phase 4: SRP Parameter Optimization 🔄 IN PROGRESS
**Goal**: Achieve <30° MAE for effective fallback

**Phase 1 Screening (100 samples)**: ✅ COMPLETED
- Tested 224 parameter combinations
- Best: 21.51° MAE / 4.50° median (n_dft_bins=32768, freq=300-4000Hz)
- Identified top 10 configurations for full testing

**Phase 2 Full Dataset Testing**: 🔄 IN PROGRESS
- Testing top 10 configurations on 2009 samples
- Expected runtime: ~6 hours per configuration
- Target: Validate <25° MAE / <10° median on full dataset

### Phase 5: Hybrid System Validation 📋 NEXT
**Objectives**:
1. Test confidence predictor on 12cm/18cm array failures
2. Validate SRP fallback effectiveness on geometric mismatch cases
3. Measure hybrid system performance:
   - CRNN accuracy on 6cm array (should maintain ~82% success rate at ≤5°)
   - SRP rescue rate on 12cm/18cm arrays (target >70% of failures with ≤5° error)
   - Overall system MAE / median across all array configurations

**Success Criteria**:
- Hybrid system MAE / median < 20° / <8° across all array sizes
- >80% success rate (≤5° error) on geometric mismatch cases
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
**Target**: Validate 21° MAE / 4.5° median performance on complete test set

## Expected Contributions

1. **Novel Discovery - Geometric Brittleness**: First comprehensive study of CRNN geometric brittleness showing:
   - Catastrophic failure (~95% failure rate at ≤5°) on full array geometry changes
   - **Gradual degradation with partial replacements**: 1-2 mic changes show progressive loss (3-16° MAE)
   - Non-linear degradation pattern (18cm more robust than 12cm despite being farther from training)
   - "Cliff edge" phenomenon: network maintains functionality up to ~2 mic replacements, then catastrophic failure
   - Position-dependent sensitivity: mic placement significantly impacts degradation severity
2. **Confidence-Based Failure Prediction**: Real-time failure prediction using CRNN's internal confidence metrics with 80% recall and 76% precision
3. **Practical Hybrid System**: Geometry-aware switching that addresses real-world deployment scenarios where array configurations vary
4. **SRP Optimization Framework**: Comprehensive parameter study identifying optimal configurations for SSL fallback (21.5° MAE / 4.5° median, 70% improvement)

## Success Metrics

1. **Hybrid System Performance**: <20° MAE / <8° median across all array geometries (6cm, 12cm, 18cm)
2. **Geometric Robustness**: >80% success rate (≤5° error) on non-training array configurations
3. **Failure Detection**: Maintain 80% recall, 76% precision for failure prediction
4. **SRP Rescue Rate**: >70% of geometric mismatch failures resolved to ≤5° error
5. **Training Geometry Preservation**: Maintain ~82% success rate on 6cm array

## Key Technical Insights

### What Works Well
- **CRNN on Training Geometry**: Excellent performance (87.8% success at ≤5°, 2.82° MAE / 2° median) when array matches training
- **CRNN with Partial 18cm Replacements**: Minimal degradation (3.34° MAE, 82% success) with single 18cm mic replacement
- **T60 Generalization**: CRNN robust to reverberation time variations (0.3s to 4.5s)
- **Confidence Metrics**: Reliable real-time failure detection with simple thresholds
- **High DFT Resolution**: 32768 bins enable precise SRP localization (21.5° MAE / 4.5° median)

### Critical Limitations
- **CRNN Full Array Geometry Change**: Catastrophic failure (95% error rate >5°) on complete array size changes
- **Progressive Geometric Degradation**: Performance degrades with number of replaced mics (1→2→8)
- **12cm More Sensitive than 18cm**: Unexpected non-linear pattern - intermediate geometry more problematic
- **Position-Dependent Sensitivity**: Specific microphone locations have varying impact on performance
- **Training Data Dependency**: CRNN learns geometry-specific acoustic patterns
- **SRP Baseline Performance**: Poor accuracy (~70° MAE / ~60° median) without optimization
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
2. ✅ **Does CRNN generalize to different array geometries?**
   - Full array change: NO - ~95% failure rate (>5°) on 12cm/18cm full arrays vs 6cm training
   - Partial replacement: PARTIALLY - gradual degradation with 1-2 mics (3-16° MAE), usable with 18cm
   - Non-linear pattern: 18cm more robust than 12cm despite being farther from training
   - Position matters: mic location significantly impacts degradation severity
3. 🔄 **Can SRP-PHAT rescue geometric mismatch failures?** Testing - optimized SRP shows 21.5° MAE / 4.5° median on samples
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
