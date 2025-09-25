#!/usr/bin/env python3
"""
Find the optimal failure predictor using ALL data - no test/train splits.
This will find the truly best threshold for practical deployment.

Usage:
    python find_best_failure_predictor.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load confidence data and prepare for analysis."""
    print("Loading confidence data...")
    df = pd.read_csv('crnn_predictions/crnn_predictions_with_confidence_clean.csv')

    # Create failure labels
    df['is_failure'] = (df['abs_error'] > 30).astype(int)

    print(f"Dataset: {len(df)} samples")
    print(f"Failures: {df['is_failure'].sum()} ({df['is_failure'].mean()*100:.1f}%)")

    return df

def find_optimal_simple_thresholds(X, y):
    """Find optimal thresholds for each feature using all data."""
    print(f"\n{'='*70}")
    print("FINDING OPTIMAL SIMPLE THRESHOLDS (ALL DATA)")
    print(f"{'='*70}")

    feature_names = X.columns
    all_results = []

    print(f"{'Feature':<20} {'F1':<6} {'Prec':<6} {'Rec':<6} {'FP':<5} {'FP%':<6} {'Threshold'}")
    print("-" * 70)

    for feature in feature_names:
        feature_values = X[feature].values

        # Try different percentile thresholds
        percentiles = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50]

        best_f1 = 0
        best_result = None

        for percentile in percentiles:
            # For entropy: higher values indicate uncertainty (failures)
            if feature == 'entropy':
                threshold = np.percentile(feature_values, 100 - percentile)
                predictions = feature_values >= threshold
                direction = '>='
            else:
                # For other features: lower values indicate uncertainty (failures)
                threshold = np.percentile(feature_values, percentile)
                predictions = feature_values <= threshold
                direction = '<='

            # Calculate metrics
            tp = np.sum(predictions & (y == 1))
            fp = np.sum(predictions & (y == 0))
            tn = np.sum(~predictions & (y == 0))
            fn = np.sum(~predictions & (y == 1))

            if tp + fp == 0 or tp + fn == 0:
                continue

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            false_positive_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

            result = {
                'feature': feature,
                'threshold': threshold,
                'direction': direction,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'false_positive_rate': false_positive_rate,
                'percentile': percentile
            }

            all_results.append(result)

            if f1 > best_f1:
                best_f1 = f1
                best_result = result

        if best_result:
            print(f"{feature:<20} {best_result['f1']:<6.3f} {best_result['precision']:<6.3f} {best_result['recall']:<6.3f} {best_result['fp']:<5} {best_result['false_positive_rate']:<6.1f} {best_result['direction']} {best_result['threshold']:.6f}")

    # Sort by F1 score
    all_results.sort(key=lambda x: x['f1'], reverse=True)
    return all_results

def analyze_100_recall_thresholds(X, y):
    """Find thresholds that achieve 100% recall using all data."""
    print(f"\n{'='*70}")
    print("100% RECALL THRESHOLDS (ALL DATA)")
    print(f"{'='*70}")

    feature_names = X.columns
    recall_100_results = []

    print(f"{'Feature':<20} {'Prec':<6} {'Rec':<6} {'FP':<5} {'FP%':<6} {'Threshold'}")
    print("-" * 70)

    for feature in feature_names:
        feature_values = X[feature].values
        failure_indices = y == 1
        failure_values = feature_values[failure_indices]

        if len(failure_values) == 0:
            continue

        # Find threshold for 100% recall
        if feature == 'entropy':
            # Higher values indicate failure
            threshold = failure_values.min() - 1e-10
            predictions = feature_values >= threshold
            direction = '>='
        else:
            # Lower values indicate failure
            threshold = failure_values.max() + 1e-10
            predictions = feature_values <= threshold
            direction = '<='

        # Calculate metrics
        tp = np.sum(predictions & (y == 1))
        fp = np.sum(predictions & (y == 0))
        tn = np.sum(~predictions & (y == 0))
        fn = np.sum(~predictions & (y == 1))

        if tp + fn == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn)
        false_positive_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

        if recall >= 0.999:  # 100% recall (allowing tiny rounding error)
            result = {
                'feature': feature,
                'threshold': threshold,
                'direction': direction,
                'precision': precision,
                'recall': recall,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'false_positive_rate': false_positive_rate
            }
            recall_100_results.append(result)
            print(f"{feature:<20} {precision:<6.3f} {recall:<6.3f} {fp:<5} {false_positive_rate:<6.1f} {direction} {threshold:.6f}")

    # Sort by precision (best precision at 100% recall)
    recall_100_results.sort(key=lambda x: x['precision'], reverse=True)
    return recall_100_results

def evaluate_ml_models_cv(X, y):
    """Evaluate ML models using cross-validation on all data."""
    print(f"\n{'='*70}")
    print("ML MODELS (5-FOLD CROSS-VALIDATION ON ALL DATA)")
    print(f"{'='*70}")

    models = {
        'SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42, C=100, gamma=0.1))
        ]),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        ),
        'XGBoost': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.2, max_depth=3, random_state=42
        ),
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'NeuralNet': Pipeline([
            ('scaler', StandardScaler()),
            ('nn', MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500))
        ])
    }

    cv_results = []

    print(f"{'Model':<20} {'F1':<6} {'±':<5} {'Prec':<6} {'Rec':<6} {'AUC':<6}")
    print("-" * 60)

    for name, model in models.items():
        try:
            # 5-fold cross-validation
            cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
            cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision')
            cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall')
            cv_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

            result = {
                'model': name,
                'cv_f1_mean': cv_f1.mean(),
                'cv_f1_std': cv_f1.std(),
                'cv_precision': cv_precision.mean(),
                'cv_recall': cv_recall.mean(),
                'cv_auc': cv_auc.mean()
            }
            cv_results.append(result)

            print(f"{name:<20} {cv_f1.mean():<6.3f} {cv_f1.std():<5.3f} {cv_precision.mean():<6.3f} {cv_recall.mean():<6.3f} {cv_auc.mean():<6.3f}")

        except Exception as e:
            print(f"{name:<20} ERROR: {str(e)[:40]}")

    cv_results.sort(key=lambda x: x['cv_f1_mean'], reverse=True)
    return cv_results

def save_results(best_f1_results, best_100_recall_results, best_ml_results):
    """Save all results to files."""

    # Save F1-optimized results
    if best_f1_results:
        f1_df = pd.DataFrame(best_f1_results)
        f1_df.to_csv('best_f1_predictors_all_data.csv', index=False)
        print(f"F1-optimized results saved to: best_f1_predictors_all_data.csv")

    # Save 100% recall results
    if best_100_recall_results:
        recall_df = pd.DataFrame(best_100_recall_results)
        recall_df.to_csv('best_100_recall_predictors_all_data.csv', index=False)
        print(f"100% recall results saved to: best_100_recall_predictors_all_data.csv")

    # Save ML results
    if best_ml_results:
        ml_df = pd.DataFrame(best_ml_results)
        ml_df.to_csv('best_ml_predictors_cv_all_data.csv', index=False)
        print(f"ML CV results saved to: best_ml_predictors_cv_all_data.csv")

def main():
    """Main execution function."""
    print("FINDING BEST FAILURE PREDICTOR - FULL DATA ANALYSIS")
    print("=" * 80)

    # Load data
    df = load_data()

    # Prepare features
    feature_cols = ['max_prob', 'entropy', 'prediction_variance', 'peak_sharpness', 'local_concentration']
    X = df[feature_cols]
    y = df['is_failure']

    print(f"Features: {feature_cols}")

    # Find optimal simple thresholds using all data
    best_f1_results = find_optimal_simple_thresholds(X, y)

    # Find 100% recall thresholds using all data
    best_100_recall_results = analyze_100_recall_thresholds(X, y)

    # Evaluate ML models with cross-validation
    best_ml_results = evaluate_ml_models_cv(X, y)

    # Summary
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATIONS")
    print(f"{'='*80}")

    if best_f1_results:
        best = best_f1_results[0]
        print(f"🎯 BEST F1-OPTIMIZED PREDICTOR:")
        print(f"   {best['feature']} {best['direction']} {best['threshold']:.8f}")
        print(f"   F1: {best['f1']:.3f}, Precision: {best['precision']:.3f}, Recall: {best['recall']:.3f}")
        print(f"   False Positives: {best['fp']}/{best['fp']+best['tn']} ({best['false_positive_rate']:.1f}%)")

    if best_100_recall_results:
        best = best_100_recall_results[0]
        print(f"\n🛡️  BEST 100% RECALL PREDICTOR:")
        print(f"   {best['feature']} {best['direction']} {best['threshold']:.8f}")
        print(f"   Precision: {best['precision']:.3f}, Recall: {best['recall']:.3f}")
        print(f"   False Positives: {best['fp']}/{best['fp']+best['tn']} ({best['false_positive_rate']:.1f}%)")

    if best_ml_results:
        best = best_ml_results[0]
        print(f"\n🤖 BEST ML MODEL (Cross-Validation):")
        print(f"   {best['model']}")
        print(f"   CV F1: {best['cv_f1_mean']:.3f}±{best['cv_f1_std']:.3f}")
        print(f"   CV Precision: {best['cv_precision']:.3f}, CV Recall: {best['cv_recall']:.3f}")

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    save_results(best_f1_results, best_100_recall_results, best_ml_results)

    print(f"\n✅ Analysis complete! Use the best thresholds above for practical deployment.")

    return {
        'best_f1': best_f1_results[0] if best_f1_results else None,
        'best_100_recall': best_100_recall_results[0] if best_100_recall_results else None,
        'best_ml': best_ml_results[0] if best_ml_results else None
    }

if __name__ == "__main__":
    results = main()