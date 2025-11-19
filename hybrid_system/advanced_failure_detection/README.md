# Advanced Failure Detection for Hybrid CRNN-SRP System

Sophisticated failure detection methods beyond simple thresholding for determining when to route CRNN predictions to SRP-PHAT backup system.

## Overview

This module implements state-of-the-art failure detection methods to address the geometric mismatch problem in sound source localization:

1. **Temperature Scaling** (Guo et al., 2017, ICML): Post-hoc calibration for better confidence estimates
2. **Mahalanobis Distance** (Lee et al., 2018, NeurIPS): Out-of-distribution detection in feature space
3. **Combined Routing**: Intelligent routing using both calibrated confidence and OOD detection

## Problem Context

- **Training**: CRNN trained on 6cm microphone array
- **Test**: System deployed with 3x12cm array (geometric mismatch)
- **Challenge**: Detect when CRNN predictions will fail and route to SRP backup
- **Goal**: Better performance than simple `max_prob < 0.04` threshold (71.8% routing accuracy)

## Module Structure

```
advanced_failure_detection/
├── __init__.py                  # Module initialization
├── extract_features.py          # Extract penultimate features from CRNN
├── temperature_scaling.py       # Calibration via temperature scaling
├── mahalanobis_ood.py          # OOD detection using Mahalanobis distance
├── combined_routing.py         # Combined routing strategy
├── run_pipeline.py             # Main orchestration script
└── README.md                   # This file
```

## Quick Start

### 1. Run Complete Pipeline

```bash
cd hybrid_system/advanced_failure_detection

# Full pipeline (all steps)
python run_pipeline.py

# With custom settings
python run_pipeline.py \
    --max_samples 1000 \
    --n_pca_components 64 \
    --output_dir results/experiment1
```

This will:
1. Extract features from CRNN for train and test sets
2. Fit temperature scaling on training data
3. Fit Mahalanobis detector on training features
4. Grid search for optimal thresholds
5. Generate visualizations and reports

### 2. Run Individual Steps

#### Extract Features

```bash
# Training set (6cm array)
python extract_features.py --split train --array_config 6cm

# Test set (6cm - in-distribution)
python extract_features.py --split test --array_config 6cm

# Test set (3x12cm - out-of-distribution)
python extract_features.py --split test --array_config 3x12cm_consecutive
```

#### Temperature Scaling

```python
from temperature_scaling import optimize_temperature, evaluate_routing_with_calibration

# Load features
train_data = np.load('features/train_6cm_features.npz', allow_pickle=True)
test_data = np.load('features/test_3x12cm_consecutive_features.npz', allow_pickle=True)

# Optimize temperature
optimal_temp, results = optimize_temperature(
    train_data['logits_pre_sig'],
    train_data['gt_angles']
)

# Evaluate routing
routing_results = evaluate_routing_with_calibration(
    test_data,
    temperature=optimal_temp,
    threshold=0.04
)
```

#### Mahalanobis OOD Detection

```python
from mahalanobis_ood import MahalanobisOODDetector, visualize_feature_space

# Initialize detector
detector = MahalanobisOODDetector(n_components=64, use_pca=True)

# Fit on training data
detector.fit(train_features, train_angles)

# Evaluate OOD detection
results = detector.evaluate_ood_detection(
    test_6cm_features,  # In-distribution
    test_3x12cm_features  # Out-of-distribution
)

# Visualize feature space
visualize_feature_space(
    train_features,
    test_6cm_features,
    test_3x12cm_features,
    output_path='feature_space_tsne.png'
)
```

#### Combined Routing

```python
from combined_routing import CombinedFailureDetector, visualize_combined_results

# Initialize and fit
detector = CombinedFailureDetector()
detector.fit(train_features, n_pca_components=64)

# Grid search for optimal thresholds
best_conf, best_dist, all_results = detector.grid_search_combined(
    test_features,
    strategy='or'  # Route if EITHER condition triggers
)

# Visualize results
best_results = [r for r in all_results if
                r['confidence_threshold'] == best_conf and
                r['distance_threshold'] == best_dist][0]
visualize_combined_results(best_results, output_dir='results')
```

## Configuration

### Microphone Array Configurations

Defined in `extract_features.py`:

```python
MIC_CONFIGS = {
    '6cm': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Training geometry
    '3x12cm_consecutive': [0, 9, 10, 11, 4, 5, 6, 7, 8],  # 3 mics replaced
    '12cm': [0] + list(range(9, 17)),  # Full 12cm array
    '18cm': [0] + list(range(17, 25)),  # Full 18cm array
    '1x12cm_pos1': [0, 9, 2, 3, 4, 5, 6, 7, 8],  # Single mic replaced
    '2x12cm_opposite': [0, 9, 2, 3, 12, 5, 6, 7, 8],  # Two opposite mics
}
```

### File Paths

Default paths in `extract_features.py`:

```python
CHECKPOINT_PATH = "/Users/danieltoberman/Documents/git/Thesis/08_CRNN/checkpoints/best_valid_loss0.0220.ckpt"
DATA_ROOT = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted"
CSV_ROOT = "/Users/danieltoberman/Documents/RealMAN_dataset_T60_08"
```

## Output Files

### Features (`.npz` files)

Each extracted feature file contains:
- `penultimate_features`: (N, T, 256) or (N, 256) - features from layer before output
- `logits_pre_sig`: (N, T, 360) or (N, 360) - raw logits before sigmoid
- `predictions`: (N, T, 360) or (N, 360) - final predictions after sigmoid
- `avg_predictions`: (N, 360) - time-averaged predictions
- `predicted_angles`: (N,) - predicted angles in degrees
- `gt_angles`: (N,) - ground truth angles
- `abs_errors`: (N,) - absolute angular errors
- `global_indices`: (N,) - original dataset indices
- `filenames`: (N,) - audio file names

### Results

After running `run_pipeline.py`:

```
results/
├── pipeline_results.npz         # Main results file
├── summary_report.txt           # Human-readable summary
├── combined_scatter.png         # Scatter: confidence vs distance
├── combined_distributions.png   # Distribution plots
└── feature_space_tsne.png      # t-SNE visualization
```

## Key Metrics

### Routing Metrics

- **Precision**: Of samples routed to SRP, what % were actual failures?
- **Recall**: Of actual failures, what % were correctly routed to SRP?
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correct routing decisions
- **Routing Rate**: % of samples routed to SRP

### Calibration Metrics

- **ECE** (Expected Calibration Error): Measures calibration quality
- **Temperature**: Learned scaling parameter (T > 1 = less confident, T < 1 = more confident)

### OOD Detection Metrics

- **AUROC**: Area under ROC curve for OOD detection
- **Mahalanobis Distance**: Distance from training distribution

## Methods

### Temperature Scaling

**Reference**: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML 2017.

- Simple post-hoc calibration method
- Learns single parameter T to scale logits: `p = sigmoid(logits / T)`
- Optimizes T on validation set to minimize NLL or ECE
- Does not change predicted angles, only confidence scores

**Advantages**:
- Extremely simple (1 parameter)
- No architectural changes
- Preserves accuracy while improving calibration

### Mahalanobis Distance

**Reference**: Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018.

- Detects out-of-distribution samples in feature space
- Fits class-conditional Gaussians on training features
- Computes Mahalanobis distance for test samples
- Uses PCA for dimensionality reduction (256 → 64 dims)

**Intuition**: Geometric mismatch causes test features to fall far from training distribution

**Advantages**:
- Directly addresses distribution shift
- No retraining required
- Interpretable (distance from training data)

### Combined Routing Strategy

Combines both methods for robust failure detection:

```
Route to SRP if:
    (calibrated_confidence < threshold_conf) OR
    (mahalanobis_distance > threshold_dist)
```

**Rationale**:
- Temperature scaling catches low-confidence predictions
- Mahalanobis distance catches OOD samples (geometric mismatch)
- OR strategy ensures we catch failures detected by either method

## Performance Expectations

Based on literature and similar problems:

- **Simple threshold** (baseline): ~70-75% routing accuracy
- **Temperature scaling alone**: +2-5% improvement in calibration, similar routing
- **Mahalanobis alone**: +5-10% improvement on geometric mismatch
- **Combined method**: +10-15% improvement, more robust

## Troubleshooting

### Out of Memory

If running out of memory during feature extraction:

```bash
python extract_features.py --split train --array_config 6cm --max_samples 5000
```

### Slow t-SNE

Reduce samples for visualization:

```python
visualize_feature_space(
    train_features,
    test_6cm_features,
    test_3x12cm_features,
    output_path='tsne.png',
    n_samples=1000  # Default is 2000
)
```

### MPS Device Issues

If MPS (Mac GPU) causes issues, use CPU:

```bash
python extract_features.py --split train --array_config 6cm --device cpu
```

## Future Extensions

Tier 2 methods documented in `research_summary.md`:

1. **ConfidNet**: Learned confidence estimation network
2. **Circular Statistics**: von Mises distribution for directional uncertainty
3. **MC Dropout**: Epistemic uncertainty via stochastic forward passes
4. **SNGP**: Spectral-Normalized Neural Gaussian Process

## Citation

If you use these methods, please cite the original papers:

```bibtex
@inproceedings{guo2017calibration,
  title={On calibration of modern neural networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q},
  booktitle={ICML},
  year={2017}
}

@inproceedings{lee2018mahalanobis,
  title={A simple unified framework for detecting out-of-distribution samples and adversarial attacks},
  author={Lee, Kimin and Lee, Kibok and Lee, Honglak and Shin, Jinwoo},
  booktitle={NeurIPS},
  year={2018}
}
```

## Contact

For questions about this implementation, refer to the main thesis documentation or research summary.
