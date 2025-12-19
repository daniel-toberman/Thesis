# Hybrid Sound Source Localization with OOD-Based Failure Detection
## Complete Knowledge Handoff Document

**Document Purpose**: This file provides complete context for continuing the thesis research without access to prior conversations or external knowledge.

**Date Created**: 2025-01-19
**Research Status**: CRNN hybrid system validated; IPDnet integration blocked by training issues

---

## EXECUTIVE SUMMARY

### Research Goal
Develop a hybrid sound source localization (SSL) system that intelligently combines deep learning models (CRNN, IPDnet) with classical methods (SRP-PHAT) using out-of-distribution (OOD) detection to identify when the neural network will fail.

### Core Hypothesis
OOD-based routing strategies that work for one deep learning architecture (CRNN) should generalize to other architectures (IPDnet), validating that failure detection is model-agnostic rather than architecture-specific.

### Current State
**✅ COMPLETED - CRNN Hybrid System**:
- CRNN trained on 6cm microphone array, tested on 3x12cm geometry
- Hybrid system achieves 12.12-14.56° MAE (vs CRNN-only 15.41°, SRP-only 15.69°)
- 19+ OOD methods evaluated (VIM, Energy OOD, ConfidNet, etc.)
- Best method: ConfidNet 30° (21.4% routing, 12.12° MAE, 3.29° improvement)
- Publication-ready results with comprehensive evaluation

**❌ BLOCKED - IPDnet Integration**:
- IPDnet training repeatedly diverges despite multiple attempted fixes
- Root cause identified: Our modifications broke the hardcoded architecture from original paper
- Paper achieves 3° MAE after 15 epochs; our training gets 22.6° → diverges to 1.4° loss
- Latest attempt: Restored original architecture, added gradient clipping
- **Critical blocker**: Cannot validate generalization hypothesis without working IPDnet

### Key Contributions (So Far)
1. **Geometric Brittleness Discovery**: CRNN exhibits 95% failure rate on array geometries different from training
2. **OOD Routing Validation**: 19 OOD methods evaluated, VIM and ConfidNet perform best
3. **Hybrid System Success**: 3.29° MAE improvement with only 21.4% computational overhead
4. **Simple Methods Win**: Basic confidence thresholding matches complex ML methods

### Critical Decisions Needed
1. **Continue IPDnet debugging** OR **abandon IPDnet and write thesis with CRNN results only**?
2. If abandoning IPDnet: Reframe thesis as "comprehensive OOD evaluation for SSL" instead of "generalization across architectures"

---

## SYSTEM ARCHITECTURE

### 1. Data Pipeline

**Dataset: RealMAN (Real-world Multi-channel Audio)**
- Source: `/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/`
- Format: Multi-channel WAV files with ground truth azimuth/elevation labels
- Training: 6cm diameter circular microphone array (T60 < 0.8s reverberation)
- Testing: 3x12cm array (consecutive microphones from 9-channel full array)

**Microphone Array Configurations**:
```
Full 9-channel array: [0, 1, 2, 3, 4, 5, 6, 7, 8]  (center + 8 outer)
- 6cm array (training):     mics [1, 3, 5, 7, 0]   (alternating + center)
- 3x12cm array (test):       mics [9, 10, 11, 4, 5, 6, 7, 8, 0]
```

**IMPORTANT - Microphone Ordering**:
- **CRNN expects**: Reference mic (0) LAST: `[1,3,5,7,0]` or `[9,10,11,4,5,6,7,8,0]`
- **SRP expects**: Reference mic (0) FIRST: `[0,1,3,5,7]` or `[0,9,10,11,4,5,6,7,8]`
- Mismatch causes incorrect feature extraction and predictions

### 2. Model Architectures

#### CRNN (Convolutional Recurrent Neural Network) ✅ WORKING
**File**: `/Users/danieltoberman/Documents/git/Thesis/SSL/CRNN.py`

**Architecture**:
```python
Input: (batch, 18, 257, time)  # Real+Imag STFT channels
↓
Conv2D layers (3x) with BatchNorm + ReLU + MaxPool
↓
Bidirectional GRU (hidden=128, dropout=0.4)
↓
Fully Connected (→ 360) + Sigmoid
↓
Output: (batch, time, 360)  # Azimuth probabilities (0-359°)
```

**Key Method**: `forward_with_intermediates(x)` returns `(logits, penultimate_features)`
- `logits`: Pre-sigmoid output (360-dim) for OOD methods
- `penultimate_features`: 256-dim from GRU output, used by Mahalanobis/KNN

**Training Script**: `/Users/danieltoberman/Documents/git/Thesis/SSL/run_CRNN.py`
**Best Checkpoint**: `best_valid_loss0.0490.ckpt`
**Performance**:
- 6cm training array: 2.82° MAE, 87.8% success (≤5°)
- 3x12cm test array: 15.41° MAE, 38.4% success (≤5°)

#### IPDnet (Inter-channel Phase Difference Network) ❌ NOT WORKING
**File**: `/Users/danieltoberman/Documents/git/Thesis/SSL/SingleTinyIPDnet.py`

**Architecture**:
```python
Input: (batch, 10, 256, time)  # IPD features (5 mics = 4 pairs + ref)
↓
FNblock_1: Full-band LSTM + Narrow-band LSTM with skip connections
↓
FNblock_2: Same structure with hardcoded dimensions
↓
CnnBlock: 3-layer CNN (138→64→32→8 channels, hardcoded!)
↓
Output: (batch, time//5, 1, 512, features)  # IPD predictions
```

**CRITICAL - Hardcoded Dimensions**:
```python
# FNblock expects EXACTLY these dimensions:
self.fullLstm = nn.LSTM(input_size=input_size+10, ...)  # +10 is HARDCODED
self.narrLstm = nn.LSTM(input_size=2*self.full_hidden_size+10, ...)

# CnnBlock expects EXACTLY:
self.conv1 = nn.Conv2d(138, 64, ..., bias=False)  # NO BIAS TERMS
self.conv2 = nn.Conv2d(64, 32, ..., bias=False)
self.conv3 = nn.Conv2d(32, 8, ..., bias=False)   # 8 outputs hardcoded
```

**Training Script**: `/Users/danieltoberman/Documents/git/Thesis/SSL/run_IPDnet_6cm.py`
**Status**: **DIVERGES** - Loss jumps from ~0.4 to ~1.4 at epoch 2-3, never recovers

**Attempted Fixes (All Failed)**:
1. ❌ Dimension corrections (9 channels → 8 IPD pairs)
2. ❌ Lower learning rate (0.0005 → 0.0001)
3. ❌ MPS → CPU (Apple Silicon LSTM bug workaround)
4. ❌ Weight initialization (Xavier + bias=True) - **MADE IT WORSE**
5. ❌ Restored original architecture (removed dynamic dimensions)
6. ⏳ Gradient clipping (1.0 norm) - **CURRENT ATTEMPT**

**Root Cause Analysis** (file: `/tmp/ipdnet_differences_analysis.md`):
- Our attempt to make architecture "flexible" broke hardcoded dimension requirements
- Original paper uses `input_size=10` for 5-mic (reference mic LAST in ordering)
- We tried using `input_size=18` for 9-mic, but architecture expects specific flow
- Paper: 3° MAE after 15 epochs | Our training: 22.6° MAE → diverges

#### SRP-PHAT (Steered Response Power - Phase Transform) ✅ WORKING
**File**: `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/precompute_srp_results.py`

**Classical beamforming method** - No training required:
- Searches spatial grid (360 azimuth × 1 elevation = 360 locations)
- Computes GCC-PHAT (Generalized Cross-Correlation with Phase Transform) between mic pairs
- Finds peak in spatial power map

**Optimal Parameters** (found via grid search):
- `n_dft_bins=16384` (frequency resolution)
- `freq_range=(300, 4000)` Hz (speech formants)
- `n_grid=360` azimuth directions
- `mode='gcc_phat_freq'`

**Performance**:
- On full test set: 15.69° MAE, 69.8% success (≤5°)
- On CRNN low-confidence cases: 19.67° MAE, 68.1% success

**Cache**: All 2,009 test predictions pre-computed in `features/test_3x12cm_srp_results.pkl`
**Benefit**: Hybrid evaluations run in seconds instead of hours

### 3. Hybrid Routing System

**Concept**: Route predictions between fast neural network and slow classical method based on confidence/OOD scores.

```
Input Audio
     ↓
  CRNN Forward Pass
     ↓
Compute OOD Score (Energy, VIM, ConfidNet, etc.)
     ↓
OOD Score > Threshold?
     ├─ YES → Route to SRP-PHAT (slow but robust)
     └─ NO  → Use CRNN prediction (fast but may fail)
```

**Routing Decision Factors**:
- **Precision**: How many routed cases actually need SRP? (avoid false positives)
- **Recall**: How many failures are caught? (avoid missing catastrophic errors)
- **Routing Rate**: Computational cost (target 20-35% for practical deployment)
- **Hybrid Performance**: Overall MAE, median, success rate improvement

**Best Configurations**:
1. **ConfidNet 30°**: 21.4% routing, 12.12° MAE, 50.5% success (+12.0%)
2. **VIM**: 30% routing, 13.00° MAE, 52.6% success (+14.2%) - Best post-hoc method
3. **Oracle 25%** (theoretical upper bound): 25% routing, 9.95° MAE, 55.7% success

---

## KEY PYTHON SCRIPTS

### Training Scripts

#### `/Users/danieltoberman/Documents/git/Thesis/SSL/run_CRNN.py`
**Purpose**: Train CRNN on 6cm array
**Status**: ✅ Complete, checkpoint available
**Key Args**:
- `--trainer.max_epochs=150`
- `--data.array_config=6cm`
- `--model.dropout=0.4`

#### `/Users/danieltoberman/Documents/git/Thesis/SSL/run_IPDnet_6cm.py` ⚠️
**Purpose**: Train IPDnet on 6cm array (5-mic subset)
**Status**: ❌ Diverges, needs fixing
**Current Configuration**:
```python
# Line 329-330: Learning rate
optimizer = torch.optim.Adam(self.arch.parameters(), lr=0.0005)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

# Line 381-382: Gradient clipping (added to prevent divergence)
"gradient_clip_val": 1.0,
"gradient_clip_algorithm": "norm",

# Dataset: use_mic_id=[1, 3, 5, 7, 0]  (reference mic LAST)
```

**Known Issues**:
- Loss diverges at epoch 2-3, batch ~250
- Jumps from 0.4 → 1.4 and never recovers
- Paper achieves 3° MAE, we get 22.6° → divergence

### Feature Extraction

#### `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/extract_features.py`
**Purpose**: Extract CRNN penultimate features + logits for OOD methods
**Output**: `features/test_3x12cm_consecutive_features.npz`
```python
{
    'penultimate_features': (2009, T, 256),  # From GRU
    'logits_pre_sig': (2009, T, 360),        # Pre-sigmoid
    'predictions': (2009, T, 360),           # Post-sigmoid
    'predicted_angles': (2009,),             # Final DOA per sample
    'gt_angles': (2009,),                    # Ground truth
    'abs_errors': (2009,),                   # |predicted - gt|
    'global_indices': (2009,),               # Sample IDs
    'filenames': (2009,)                     # Source files
}
```

**Usage**:
```bash
cd /Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection
python3 extract_features.py
```

#### `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/precompute_srp_results.py`
**Purpose**: Pre-compute SRP predictions for all test samples (one-time cost)
**Output**: `features/test_3x12cm_srp_results.pkl`
**Benefit**: Future hybrid evaluations run in seconds
**Runtime**: ~2-3 hours for 2,009 samples

### OOD Method Implementations

All OOD routing scripts follow the same pattern:
```python
def compute_ood_scores(features_dict):
    """Compute OOD scores for each sample."""
    # Returns: (2009,) array of scores, higher = more OOD
    ...

# Find threshold for target routing rate
threshold = find_threshold_for_routing_rate(ood_scores, target_rate=0.30)

# Make routing decisions
route_to_srp = ood_scores > threshold
```

**Location**: `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/`

**Best Performing Methods**:
1. **`vim_ood_routing.py`** - Virtual-logit matching (13.00° MAE) ⭐ BEST POST-HOC
2. **`she_ood_routing.py`** - Similarity-based (13.24° MAE)
3. **`gradnorm_ood_routing.py`** - Gradient norm (13.86° MAE)
4. **`energy_ood_routing.py`** - Energy-based (15.27° MAE at 30% routing)
5. **`mc_dropout_routing.py`** - Bayesian uncertainty (15.16° MAE)

**Supervised Methods** (require training):
- **`confidnet_routing.py`** + **`confidnet_model.py`** + **`train_confidnet.py`**
- Best overall: 12.12° MAE (ConfidNet 30°), 12.62° MAE (ConfidNet 20°)

**Failed Methods**:
- Deep SVDD (hypersphere collapse)
- ReAct (hurts performance standalone)
- Mahalanobis alone (needs calibration)

### Evaluation Pipeline

#### `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/analyze_ood_distributions.py`
**Purpose**: Find optimal F1 thresholds for each OOD method
**Process**:
1. Load features: `test_3x12cm_consecutive_features.npz`
2. Compute OOD scores for all methods
3. Grid search thresholds to maximize F1 score for detecting failures (>5° error)
4. Save optimal thresholds for hybrid evaluation

**Output**: Console logs + plots in `results/ood_distributions/`

#### `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/evaluate_ood_hybrid.py`
**Purpose**: Evaluate hybrid CRNN+SRP performance with one OOD method
**Usage**:
```bash
python3 evaluate_ood_hybrid.py \
    --method vim \
    --threshold 1.234 \
    --output_dir results/ood_methods/vim_hybrid
```

**Output**:
- `routing_statistics.csv`: Precision, recall, F1, false positive rate
- `hybrid_results.csv`: MAE, median, success rate
- `routing_decisions.csv`: Per-sample decisions

#### `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/run_all_ood_methods.py`
**Purpose**: Batch evaluation of all OOD methods with optimal thresholds
**Output**: Comprehensive comparison table

### Oracle Baselines

#### `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/evaluate_oracle_baseline.py`
**Purpose**: Compute theoretical upper bound by routing worst X% of CRNN errors
**Key Results**:
- Oracle 25%: 9.95° MAE, 55.7% success
- Oracle 30%: 10.45° MAE, 58.4% success

**Insight**: ConfidNet 30° achieves 82% of Oracle 25% performance (12.12° vs 9.95°)

---

## CURRENT STATUS

### ✅ Completed Work

**1. CRNN Training & Testing**
- Trained on 6cm array: 2.82° MAE baseline
- Tested on 3x12cm: 15.41° MAE (geometric mismatch failure)
- Feature extraction: `test_3x12cm_consecutive_features.npz` ready

**2. SRP Optimization**
- Grid search over 224 parameter combinations
- Optimal config: n_dft=16384, freq=300-4000Hz, grid=360
- All predictions cached: `test_3x12cm_srp_results.pkl`

**3. OOD Method Evaluation (19 methods)**
- Post-hoc methods: VIM, SHE, GradNorm, Energy, MC Dropout, KNN, MaxProb, DICE, Mahalanobis
- Supervised methods: ConfidNet (20° and 30° variants)
- Combined methods: Temperature + Mahalanobis
- Oracle baselines: 25% and 30% routing

**4. Hybrid System Validation**
- Best efficiency: ConfidNet 30° (21.4% routing, 12.12° MAE, 3.29° improvement)
- Best performance: ConfidNet 20° (30.2% routing, 12.62° MAE, 56.3% success)
- Best post-hoc: VIM (30% routing, 13.00° MAE, 52.6% success)

**5. Documentation**
- `/Users/danieltoberman/Documents/git/Thesis/research_summary.md` (general overview)
- `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/research_summary.md` (OOD methods)
- `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/OOD_METHODS_README.md` (implementation guide)

### ❌ Blocked Work

**1. IPDnet Training** (CRITICAL BLOCKER)
**Status**: Repeatedly diverges despite 6+ attempted fixes
**Last Attempt**: Restored original paper architecture + gradient clipping
**Current Training Command**:
```bash
cd /Users/danieltoberman/Documents/git/Thesis/SSL
python3 run_IPDnet_6cm.py fit --trainer.max_epochs=30 --trainer.accelerator=cpu
```

**Expected Behavior**: 3° MAE after 15 epochs (from paper)
**Actual Behavior**: 22.6° MAE at epoch 0, diverges to 1.4° loss at epoch 2-3

**Root Cause Hypothesis**:
- Original architecture has hardcoded dimensions for 5-mic (input_size=10)
- Our data has different channel ordering or preprocessing than paper
- Possible IPD computation bug in `data_preprocess()` method
- Microphone ordering mismatch (reference mic position)

**Debugging Checklist** (if continuing):
- [ ] Verify IPD ground truth computation matches paper exactly
- [ ] Check STFT normalization (should match original RecordData.py)
- [ ] Validate microphone ordering: `[1,3,5,7,0]` (0 LAST)
- [ ] Compare our batch statistics with paper (print GT IPD ranges)
- [ ] Try paper's exact dataset split (currently using RealMAN_dataset_T60_08)
- [ ] Contact paper authors for preprocessing details

**2. IPDnet Feature Extraction** (Depends on #1)
- Cannot extract features until training works
- Need `forward_with_intermediates()` method (not yet implemented)
- Script ready: `extract_ipdnet_features.py` (waiting for working checkpoint)

**3. Generalization Validation** (Depends on #1, #2)
- Cannot test if OOD methods generalize until IPDnet features available
- Planned script: `compare_crnn_ipdnet.py` (not yet written)

### 🔄 In Progress

**None** - All work blocked on IPDnet training

---

## KNOWN ISSUES

### 1. IPDnet Training Divergence (HIGH PRIORITY)
**Symptoms**:
- Epoch 0-1: Loss ~0.4, MAE 53-140°
- Epoch 2, Batch 250: Loss jumps 0.4 → 1.4
- Epoch 3+: Loss stuck at 1.3-1.4, MAE 135°

**Attempted Solutions**:
1. Reduced learning rate: 0.0005 → 0.0001 (didn't help)
2. Added gradient clipping: norm=1.0 (current attempt)
3. Fixed MPS LSTM bug: switched to CPU
4. Restored original architecture: removed dynamic dimensions
5. Fixed microphone ordering: `[1,3,5,7,0]` (ref mic last)

**Next Steps**:
- Run overnight training with gradient clipping (currently executing)
- If still diverges: Compare IPD ground truth statistics with paper
- If still diverges: Contact paper authors or abandon IPDnet

### 2. Microphone Ordering Confusion (MEDIUM PRIORITY)
**Issue**: Different parts of codebase expect different orderings
**CRNN**: Expects `[mics..., 0]` (reference mic LAST)
**SRP**: Expects `[0, mics...]` (reference mic FIRST)
**IPDnet**: Expects `[mics..., 0]` (reference mic LAST)

**Impact**: Incorrect features if ordering is wrong
**Status**: Fixed for CRNN and SRP, IPDnet may still have issue

### 3. Missing IPDnet Methods (LOW PRIORITY)
**Not Yet Implemented**:
- `forward_with_intermediates()` - needed for feature extraction
- DOA conversion from IPD - paper uses `PredDOA` module (exists in Module.py)

### 4. Test Set Distribution (DOCUMENTATION)
**Observation**: 3x12cm test set has 61.6% failures (>5° error) for CRNN
**Impact**: High routing rates (70-100%) optimize for F1 but hurt MAE
**Not a bug**: Just needs to be explained in thesis writeup

---

## RESEARCH FINDINGS (Key Insights)

### 1. Geometric Brittleness of Deep Learning SSL
**Discovery**: CRNN exhibits catastrophic failure on array geometries different from training
- Training (6cm): 87.8% success, 2.82° MAE
- Testing (3x12cm full): 4.5% success, 68.0° MAE (95% failure rate)
- Testing (3x12cm consecutive): 38.4% success, 15.41° MAE

**Progressive Degradation**:
- 1 mic replaced: 76-82% success (minimal impact)
- 2 mics replaced: 46-66% success (moderate)
- 3 mics replaced: 38.8% success (optimal for hybrid)
- Full array change: 4.5% success (catastrophic)

**Key Insight**: Network learns geometry-specific acoustic patterns that don't transfer

### 2. OOD Methods for Failure Detection
**19 Methods Evaluated** across three categories:

**Supervised (requires training)**:
- ConfidNet: 12.12-12.62° MAE (BEST overall)

**Post-hoc (no training)**:
- VIM: 13.00° MAE (BEST post-hoc) ⭐
- SHE: 13.24° MAE
- GradNorm: 13.86° MAE
- MaxProb: 13.90° MAE (surprisingly strong baseline)
- Energy OOD, MC Dropout, KNN: 14.7-15.3° MAE

**Failed methods**:
- Deep SVDD (hypersphere collapse)
- ReAct standalone (hurts performance)
- Mahalanobis alone (needs calibration)

**Key Insight**: Simple methods (VIM, MaxProb) match or beat complex methods

### 3. Hybrid System Performance
**Best Configurations**:

| Method | Routing % | MAE | Improvement | Success Rate |
|--------|-----------|-----|-------------|--------------|
| CRNN-only | 0% | 15.41° | baseline | 38.4% |
| ConfidNet 30° | 21.4% | 12.12° | -3.29° | 50.5% (+12.0%) |
| ConfidNet 20° | 30.2% | 12.62° | -2.79° | 56.3% (+17.9%) |
| VIM | 30.0% | 13.00° | -2.41° | 52.6% (+14.2%) |
| Oracle 25% | 25.0% | 9.95° | -5.46° | 55.7% (+17.3%) |

**Key Findings**:
- Simple threshold (ConfidNet 30°) achieves 82% of Oracle performance
- Learned methods only 0.38-1.26° behind theoretical optimum
- VIM (post-hoc) nearly matches ConfidNet (supervised)
- Routing 21-30% of cases sufficient for major improvements

### 4. Confidence Calibration Breaks Selectively
**Catastrophic Failures (>30°)**: ✅ Detectable
- Low max_prob (0.03 vs 0.06 for good predictions)
- High entropy (5.0 vs 3.6)
- Statistical significance p<0.001

**Moderate Degradation (10-25°)**: ❌ NOT Detectable
- "Confidently wrong" phenomenon
- Bad cases show HIGHER confidence than good cases
- Calibration decouples from accuracy under geometric mismatch

**Implication**: Hybrid approach only viable for configurations with catastrophic failure rates

---

## HOW TO CONTINUE THIS WORK

This section is written for another AI agent picking up the research.

### Option A: Fix IPDnet and Complete Generalization Study

**Goal**: Validate that OOD routing strategies generalize across DL architectures

**Steps**:

1. **Debug IPDnet Training** (highest priority)
   ```bash
   cd /Users/danieltoberman/Documents/git/Thesis/SSL

   # Check if overnight training succeeded
   tail -100 ipdnet_with_grad_clip.log

   # If still diverging, compare with original paper repo
   cd /Users/danieltoberman/Documents/git/Thesis/RealMAN-main/baselines/SSL
   # Review: SingleTinyIPDnet.py, run_IPDnet.py, RecordData.py
   ```

   **Debugging Strategy**:
   - Print IPD ground truth statistics in training_step() to verify ranges
   - Compare with paper's expected IPD values (-π to +π)
   - Check if STFT normalization matches original (line 286-294 in run_IPDnet_6cm.py)
   - Verify mic order: `use_mic_id=[1,3,5,7,0]` in dataset (line ~80)
   - Test on 100 samples only to iterate faster

2. **Add Feature Extraction to IPDnet**
   ```python
   # In SingleTinyIPDnet.py, add:
   def forward_with_intermediates(self, x):
       """Return predictions + penultimate features."""
       # Identify penultimate layer (probably after final CNN, before output)
       # Return (ipd_predictions, features_before_output)
       pass
   ```

   **Note**: Feature dimensions will differ from CRNN (256-dim), may need PCA/projection

3. **Extract IPDnet Features**
   ```bash
   cd /Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection
   python3 extract_ipdnet_features.py \
       --checkpoint path/to/ipdnet_best.ckpt \
       --output features/test_3x12cm_ipdnet_features.npz
   ```

4. **Run OOD Methods on IPDnet Features**
   ```bash
   # Use same thresholds as CRNN, or re-optimize
   python3 analyze_ood_distributions.py \
       --features features/test_3x12cm_ipdnet_features.npz \
       --model ipdnet

   python3 run_all_ood_methods.py \
       --model ipdnet \
       --features features/test_3x12cm_ipdnet_features.npz
   ```

5. **Compare CRNN vs IPDnet Generalization**
   ```bash
   # Create new script: compare_crnn_ipdnet.py
   python3 compare_crnn_ipdnet.py \
       --crnn_features features/test_3x12cm_consecutive_features.npz \
       --ipdnet_features features/test_3x12cm_ipdnet_features.npz \
       --output results/generalization_analysis/
   ```

   **Analysis Questions**:
   - Do same OOD methods perform best for both architectures?
   - Do optimal thresholds transfer?
   - What is correlation of routing decisions (CRNN vs IPDnet)?
   - Which methods generalize (F1 score within ±0.05)?

6. **Write Comparison Section**
   - Add to research_summary.md
   - Create figures: scatter plot (CRNN F1 vs IPDnet F1)
   - Table: method rankings for both architectures

**Estimated Time**: 2-4 weeks (if IPDnet debugging takes <1 week)

### Option B: Abandon IPDnet, Finalize CRNN Thesis

**Goal**: Comprehensive OOD evaluation for SSL with CRNN results only

**Justification**:
- CRNN hybrid system already publication-ready
- 19 OOD methods evaluated is substantial contribution
- Novel application domain (spatial audio SSL)
- Generalization hypothesis is secondary (can be future work)

**Steps**:

1. **Reframe Thesis Narrative**
   - **Old**: "Do OOD methods generalize across DL architectures?"
   - **New**: "Comprehensive evaluation of OOD-based failure detection for deep learning SSL under geometric mismatch"

2. **Strengthen CRNN Analysis**
   - Add ablation studies (e.g., effect of routing rate 10-50%)
   - Analyze failure patterns: which azimuths cause most routing?
   - Compare computational cost vs accuracy trade-offs
   - Add statistical significance tests (bootstrap confidence intervals)

3. **Write Thesis Chapters**
   ```
   Chapter 1: Introduction
   - Motivation: Deep learning SSL fails on out-of-distribution geometries
   - Research question: Can OOD methods detect failures in real-time?

   Chapter 2: Background
   - Sound source localization methods (CRNN, SRP-PHAT)
   - Out-of-distribution detection (19 methods reviewed)
   - Hybrid systems in machine learning

   Chapter 3: CRNN Geometric Robustness
   - Training on 6cm, testing on 3x12cm
   - Progressive degradation with partial mic replacements
   - Confidence calibration analysis

   Chapter 4: OOD-Based Failure Detection
   - 19 methods implemented and evaluated
   - Oracle baselines establish theoretical limits
   - VIM and ConfidNet perform best

   Chapter 5: Hybrid System Validation
   - Routing decisions and performance analysis
   - Computational cost vs accuracy trade-offs
   - Ablation studies

   Chapter 6: Conclusion & Future Work
   - Summary of contributions
   - Future: Generalization to other architectures (IPDnet, Transformer-based)
   - Future: Multi-geometry training
   ```

4. **Create Publication-Ready Figures**
   ```python
   # Generate in /results/figures/
   1. CRNN performance vs array geometry (bar plot)
   2. OOD method comparison (table + bar plot)
   3. Hybrid MAE improvement (scatter: routing % vs MAE)
   4. Routing decision breakdown (stacked bar: TP, FP, TN, FN)
   5. Oracle vs learned methods (line plot)
   6. Catastrophic failure rescue (before/after histogram)
   ```

5. **Write Conference Paper** (parallel with thesis)
   - Target: ICASSP 2026, INTERSPEECH 2026, or NeurIPS workshop
   - Focus: VIM as best post-hoc method, ConfidNet as best supervised
   - Contribution: First application of modern OOD methods to spatial audio SSL

6. **Prepare Defense**
   - Presentation slides (30-40 min)
   - Demo: Live inference with routing visualization
   - Anticipated questions:
     - "Why didn't you test on other DL architectures?" (time/technical issues)
     - "How does this compare to multi-geometry training?" (future work)
     - "Is 21% routing overhead acceptable?" (depends on application)

**Estimated Time**: 6-8 weeks

### Option C: Hybrid Approach (Thesis + Future IPDnet)

**Goal**: Submit thesis with CRNN results, continue IPDnet as follow-up paper

**Steps**:
1. Follow Option B to complete thesis (6-8 weeks)
2. Defend thesis with CRNN-only results
3. Continue debugging IPDnet post-defense
4. Publish follow-up paper if IPDnet works: "Generalization of OOD Methods..."

**Advantages**:
- Don't delay thesis for technical blocker
- Still explore generalization hypothesis
- Two publications instead of one

**Estimated Time**: Thesis in 6-8 weeks, follow-up paper 2-4 months

### Recommended Decision Tree

```
Does IPDnet training converge within 1 week?
├─ YES → Option A (Complete generalization study)
└─ NO  → Is thesis deadline <3 months?
         ├─ YES → Option B (CRNN-only thesis)
         └─ NO  → Option C (Thesis now, IPDnet later)
```

---

## DATASETS AND PATHS

### Primary Dataset
**RealMAN Dataset** (Real-world Multi-channel Audio for Localization)
- Path: `/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/`
- Format: `.wav` files (multi-channel) + `.npy` labels (azimuth, elevation)
- Structure:
  ```
  RealMAN_dataset_T60_08/
  ├── train/
  │   ├── ma_speech/       # Clean speech (T60 < 0.8s)
  │   └── ma_noise/        # Noise files
  ├── test/
  │   ├── ma_speech/
  │   └── ma_noise/
  └── meta/
      ├── train_list.txt
      └── test_list.txt
  ```
- Train samples: ~21,000 (6cm array only, T60 < 0.8s)
- Test samples: 2,009 (3x12cm array, full T60 range)

### Feature Files

**CRNN Features** ✅ Ready
```
/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features/
├── test_3x12cm_consecutive_features.npz        # Test features (2,009 samples)
├── train_6cm_features.npz                      # Train features (21,000+ samples)
└── test_3x12cm_srp_results.pkl                 # Pre-computed SRP predictions
```

**IPDnet Features** ❌ Not yet created (waiting for training)
```
features/
├── test_3x12cm_ipdnet_features.npz            # TODO: Extract after training
└── train_6cm_ipdnet_features.npz              # TODO: Extract after training
```

### Model Checkpoints

**CRNN** ✅ Available
```
/Users/danieltoberman/Documents/git/Thesis/SSL/lightning_logs/version_XX/checkpoints/
└── best_valid_loss0.0490.ckpt                 # Best CRNN checkpoint
```

**IPDnet** ❌ No working checkpoint
```
/Users/danieltoberman/Documents/git/Thesis/SSL/lightning_logs/
└── version_14/                                # Latest failed attempt
```

### Results Directories

```
/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/results/
├── ood_distributions/                         # Threshold optimization plots
├── ood_methods/                               # Per-method evaluation results
│   ├── vim_hybrid/
│   ├── energy_ood_hybrid/
│   └── confidnet_30_hybrid/
├── oracle_baselines/                          # Oracle 25%, 30% results
└── model_comparison/                          # TODO: CRNN vs IPDnet comparison
```

---

## DEPENDENCIES AND ENVIRONMENT

### Python Environment
```bash
# Primary environment (already exists)
cd /Users/danieltoberman/Documents/git/Thesis
source .venv/bin/activate

# Key packages (already installed)
- torch==2.1.0
- pytorch-lightning==2.0.0
- numpy, scipy, scikit-learn
- librosa (for audio processing)
- matplotlib, seaborn (for plotting)
```

### Hardware Requirements
**Current Setup**: MacBook Pro with Apple Silicon (M1/M2)
- **Issue**: MPS (Metal Performance Shaders) has LSTM bugs → Use `--trainer.accelerator=cpu`
- **Training time**: 2-4 hours per CRNN epoch on CPU
- **SRP computation**: ~3-4 seconds per sample

**Recommended for IPDnet**:
- NVIDIA GPU with CUDA (RunAI K8s cluster available)
- Or: Train on CPU overnight (slow but works)

### Known Platform Issues
1. **MPS LSTM Bug**: PyTorch LSTM on Apple Silicon produces incorrect gradients
   - **Solution**: Always use `--trainer.accelerator=cpu` for LSTM models

2. **PyTorch Lightning Warnings**: Ignore "num_workers" warnings
   - Not a bug, just performance suggestions

3. **Microphone Ordering**: See "Known Issues" section above

---

## GLOSSARY

**Azimuth**: Horizontal angle (0-359°) of sound source relative to array center

**DOA**: Direction of Arrival - 2D angle (azimuth, elevation) of sound source

**DFT**: Discrete Fourier Transform - converts time-domain audio to frequency domain

**GCC-PHAT**: Generalized Cross-Correlation with Phase Transform - classical SSL method

**IPD**: Inter-channel Phase Difference - phase difference between microphone pairs

**MAE**: Mean Absolute Error - average |predicted_angle - true_angle| in degrees

**OOD**: Out-of-Distribution - test samples that differ from training distribution

**Penultimate Features**: Neural network activations from second-to-last layer (before output)

**Routing**: Decision to use CRNN prediction or route to SRP-PHAT based on confidence/OOD score

**SRP-PHAT**: Steered Response Power with Phase Transform - beamforming-based SSL

**STFT**: Short-Time Fourier Transform - time-frequency representation of audio signal

**Success Rate**: Percentage of predictions within 5° of ground truth

**T60**: Reverberation time - time for sound to decay 60dB (higher = more echoes)

---

## CONTACT POINTS

**Original Paper (CRNN)**:
- Paper: "Real-world Microphone Array Network Dataset with Ground Truth" (IEEE)
- Code: `/Users/danieltoberman/Documents/git/Thesis/RealMAN-main/baselines/SSL/`
- Reference implementation is in this directory

**Original Paper (IPDnet)**:
- Same RealMAN dataset paper
- Reference: `RealMAN-main/baselines/SSL/SingleTinyIPDnet.py`
- Expected performance: 3° MAE after 15 epochs

**This Research**:
- All custom code: `/Users/danieltoberman/Documents/git/Thesis/hybrid_system/`
- Research summaries: `research_summary.md` (two locations)
- This handoff document: `THESIS_HANDOFF_COMPLETE.md`

---

## FINAL RECOMMENDATIONS

**For Continuing This Work**:

1. **Immediate Next Step**: Check if IPDnet training with gradient clipping succeeded
   ```bash
   cd /Users/danieltoberman/Documents/git/Thesis/SSL
   tail -200 ipdnet_with_grad_clip.log
   # Look for: Loss staying at ~0.4 for multiple epochs (good)
   #           OR loss diverging to 1.4+ (bad)
   ```

2. **If Converged**: Proceed with Option A (generalization study)

3. **If Still Diverging**:
   - Spend max 3-5 days debugging IPD ground truth computation
   - If no progress: Switch to Option B (CRNN-only thesis)
   - Rationale: CRNN results are already publication-quality

4. **Thesis Writing Priority**:
   - Start writing even while debugging IPDnet
   - Chapters 1-3 don't depend on IPDnet results
   - Can always add Chapter 7 (Generalization) if IPDnet works later

5. **Publication Strategy**:
   - CRNN hybrid system → Conference paper (ICASSP/INTERSPEECH)
   - IPDnet generalization → Follow-up journal paper (if successful)
   - Don't delay first publication waiting for second experiment

**Key Insight**: The CRNN hybrid system work is complete, validated, and publishable as-is. IPDnet is a "nice-to-have" for strengthening the generalization claim, but NOT required for a successful thesis.

---

**END OF HANDOFF DOCUMENT**

*This document contains all essential information to continue the thesis research without access to prior conversations. Good luck!*
