# Hybrid SSL System - CRNN Failure Detection & SRP Optimization

This folder contains the core components for the hybrid sound source localization system that combines CRNN neural networks with classical SRP-PHAT methods.

## Directory Structure

### `/failure_prediction/`
Core failure detection system - predicts when CRNN will fail and switches to SRP backup:
- `analyze_confidence_metrics.py` - Analyzes CRNN confidence metrics for failure prediction
- `find_best_failure_predictor.py` - Finds optimal thresholds/models for failure detection
- `train_failure_predictor.py` - Trains ML models for failure prediction
- `test_srp_on_predicted_failures.py` - Tests hybrid system performance

**Key Achievement**: 78.1% F1 score for failure prediction (80.1% recall, 76.1% precision)

### `/srp_optimization/`
SRP-PHAT optimization for challenging failure cases:
- `advanced_srp_methods.py` - Advanced SRP processing with multiple optimization techniques
- `optimize_srp_parameters.py` - Grid search optimization for SRP parameters

**Key Achievement**: Ensemble SRP achieves 26.9% success rate on CRNN failure cases (vs 23.9% CRNN baseline)

### `/results/`
Important result files from experiments:
- `srp_predicted_failures_test.csv` - CRNN failure cases identified by confidence predictor
- `srp_predicted_failures_test_srp_results.csv` - SRP results on predicted failure cases
- `best_f1_predictors_all_data.csv` - Optimal F1 failure predictors
- `best_100_recall_predictors_all_data.csv` - High recall failure predictors
- `best_ml_predictors_cv_all_data.csv` - Cross-validated ML failure predictors

### `/analysis/`
Analysis results and intermediate data:
- `confidence_analysis/` - CRNN confidence metric analysis results
- `failure_predictor_results/` - Failure prediction model evaluation results
- `crnn_predictions/` - CRNN prediction outputs with confidence scores

## Usage

### Run Failure Detection
```bash
python failure_prediction/find_best_failure_predictor.py
```

### Test Hybrid System
```bash
python failure_prediction/test_srp_on_predicted_failures.py simple
```

### Optimize SRP Parameters
```bash
python srp_optimization/optimize_srp_parameters.py
```

## Key Findings

1. **Failure Prediction**: CRNN failures can be predicted with 80.1% recall using max_prob ≤ 0.02560333
2. **Automotive Failures**: 99.5% of CRNN failures occur in automotive environments with novel noise
3. **Hybrid Performance**: Ensemble SRP beats CRNN baseline by 3.0% on predicted failure cases
4. **Array Size**: Larger microphone arrays (12cm, 18cm) provide no improvement over 6cm baseline

## Research Impact

This hybrid system addresses the specific failure modes of neural SSL methods by:
- Real-time failure detection using network confidence
- Automatic fallback to optimized classical methods
- Maintaining excellent performance in normal conditions while rescuing challenging cases