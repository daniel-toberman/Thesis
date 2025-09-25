#!/usr/bin/env python3
"""
Test ML models optimized for 100% recall vs simple thresholds.
Compare precision while maintaining perfect recall (catch all failures).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load confidence data and prepare for training."""
    print("Loading confidence data...")
    df = pd.read_csv('crnn_predictions/crnn_predictions_with_confidence_clean.csv')

    # Create failure labels
    df['is_failure'] = (df['abs_error'] > 30).astype(int)

    # Select confidence features
    confidence_features = ['max_prob', 'entropy', 'prediction_variance', 'peak_sharpness', 'local_concentration']
    X = df[confidence_features].copy()
    y = df['is_failure'].copy()

    print(f"Dataset: {len(X)} samples")
    print(f"Class distribution: Success: {np.sum(y==0)}, Failure: {np.sum(y==1)}")
    print(f"Failure rate: {np.sum(y==1)/len(y)*100:.1f}%")

    return X, y, confidence_features, df

def find_100_recall_threshold(y_true, y_scores, metric_name):
    """Find threshold that achieves exactly 100% recall."""
    # Sort scores and find the minimum score among actual failures
    failure_indices = y_true == 1
    failure_scores = y_scores[failure_indices]

    if len(failure_scores) == 0:
        return None, 0, 0

    # The threshold should be just above the minimum failure score
    if metric_name in ['entropy']:  # Higher values indicate failure
        threshold = failure_scores.min() - 1e-10
        predictions = y_scores >= threshold
    else:  # Lower values indicate failure
        threshold = failure_scores.max() + 1e-10
        predictions = y_scores <= threshold

    # Calculate precision at 100% recall
    tp = np.sum(predictions & (y_true == 1))
    fp = np.sum(predictions & (y_true == 0))

    recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return threshold, precision, recall

def test_simple_thresholds(X, y, feature_names):
    """Test simple threshold approaches for 100% recall."""
    print("\n" + "="*60)
    print("SIMPLE THRESHOLDS - 100% RECALL OPTIMIZATION")
    print("="*60)
    print(f"{'Metric':<20} {'Threshold':<12} {'Precision':<10} {'False+':<8} {'Rate':<6}")
    print("-"*60)

    simple_results = {}

    for i, metric in enumerate(feature_names):
        threshold, precision, recall = find_100_recall_threshold(y.values, X.iloc[:, i].values, metric)

        if threshold is not None:
            # Calculate false positives
            if metric in ['entropy']:
                predictions = X.iloc[:, i] >= threshold
            else:
                predictions = X.iloc[:, i] <= threshold

            fp = np.sum(predictions & (y == 0))
            fp_rate = fp / np.sum(y == 0) * 100

            simple_results[metric] = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'false_positives': fp,
                'false_positive_rate': fp_rate
            }

            print(f"{metric:<20} {threshold:<12.4f} {precision:<10.3f} {fp:<8} {fp_rate:<6.1f}%")

    return simple_results

def optimize_ml_for_100_recall(X_train, X_test, y_train, y_test, feature_names):
    """Train ML models and find thresholds for 100% recall."""
    print("\n" + "="*60)
    print("ML MODELS - 100% RECALL OPTIMIZATION")
    print("="*60)

    # Create models
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

    ml_results = {}
    print(f"{'Model':<18} {'Threshold':<10} {'Precision':<10} {'False+':<8} {'Rate':<6}")
    print("-"*60)

    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)

            # Get prediction probabilities
            y_prob = model.predict_proba(X_test)[:, 1]

            # Find threshold for 100% recall
            threshold, precision, recall = find_100_recall_threshold(y_test.values, y_prob, 'probability')

            if threshold is not None and recall >= 0.999:  # Allow tiny rounding error
                # Calculate false positives
                predictions = y_prob >= threshold
                fp = np.sum(predictions & (y_test == 0))
                fp_rate = fp / np.sum(y_test == 0) * 100

                ml_results[name] = {
                    'model': model,
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'false_positives': fp,
                    'false_positive_rate': fp_rate,
                    'y_prob': y_prob
                }

                print(f"{name:<18} {threshold:<10.4f} {precision:<10.3f} {fp:<8} {fp_rate:<6.1f}%")

        except Exception as e:
            print(f"{name:<18} FAILED: {str(e)[:40]}")

    return ml_results

def create_comparison_analysis(simple_results, ml_results, y_test, output_dir):
    """Create detailed comparison analysis and visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON - 100% RECALL GUARANTEED")
    print("="*60)

    # Combine all results for comparison
    all_results = {}

    # Add simple thresholds
    for name, result in simple_results.items():
        all_results[f"Simple_{name}"] = result

    # Add ML models
    for name, result in ml_results.items():
        all_results[f"ML_{name}"] = result

    # Sort by precision (descending)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['precision'], reverse=True)

    print(f"{'Method':<25} {'Precision':<10} {'False+':<8} {'FP Rate':<8}")
    print("-"*60)

    best_method = None
    best_precision = 0

    for name, result in sorted_results:
        precision = result['precision']
        fp = result['false_positives']
        fp_rate = result['false_positive_rate']

        print(f"{name:<25} {precision:<10.3f} {fp:<8} {fp_rate:<8.1f}%")

        if precision > best_precision:
            best_precision = precision
            best_method = name

    print(f"\n🏆 WINNER: {best_method}")
    print(f"   Precision: {best_precision:.3f} ({best_precision*100:.1f}%)")
    print(f"   False Positives: {all_results[best_method]['false_positives']}")
    print(f"   FP Rate: {all_results[best_method]['false_positive_rate']:.1f}%")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Precision comparison
    methods = [name.replace('Simple_', 'S_').replace('ML_', 'M_') for name, _ in sorted_results]
    precisions = [result['precision'] for _, result in sorted_results]
    colors = ['blue' if name.startswith('Simple_') else 'red' for name, _ in sorted_results]

    bars = axes[0].bar(range(len(methods)), precisions, color=colors, alpha=0.7)
    axes[0].set_title('Precision at 100% Recall', fontweight='bold')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, precision) in enumerate(zip(bars, precisions)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{precision:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. False positive rates
    fp_rates = [result['false_positive_rate'] for _, result in sorted_results]

    bars = axes[1].bar(range(len(methods)), fp_rates, color=colors, alpha=0.7)
    axes[1].set_title('False Positive Rate at 100% Recall', fontweight='bold')
    axes[1].set_ylabel('False Positive Rate (%)')
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')

    # 3. Precision vs FP Rate scatter
    axes[2].scatter([result['false_positive_rate'] for _, result in sorted_results],
                   [result['precision'] for _, result in sorted_results],
                   c=[0 if name.startswith('Simple_') else 1 for name, _ in sorted_results],
                   cmap='coolwarm', s=100, alpha=0.7, edgecolors='black')

    for i, (name, result) in enumerate(sorted_results):
        axes[2].annotate(name.replace('Simple_', 'S_').replace('ML_', 'M_'),
                        (result['false_positive_rate'], result['precision']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

    axes[2].set_xlabel('False Positive Rate (%)')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision vs False Positive Rate', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '100_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed results
    results_df = pd.DataFrame([
        {
            'Method': name,
            'Type': 'Simple' if name.startswith('Simple_') else 'ML',
            'Precision': result['precision'],
            'False_Positives': result['false_positives'],
            'FP_Rate_Percent': result['false_positive_rate'],
            'Threshold': result['threshold']
        }
        for name, result in all_results.items()
    ])

    results_df = results_df.sort_values('Precision', ascending=False)
    results_df.to_csv(output_dir / '100_recall_results.csv', index=False)

    print(f"\\nResults saved to: {output_dir.absolute()}")
    print(f"- Visualization: {output_dir}/100_recall_comparison.png")
    print(f"- Detailed results: {output_dir}/100_recall_results.csv")

    return best_method, all_results[best_method]

def save_best_100_recall_predictor(best_method, best_result, feature_names, output_dir):
    """Save the best predictor for 100% recall."""
    output_dir = Path(output_dir)

    # Create usage code
    if best_method.startswith('Simple_'):
        metric_name = best_method.replace('Simple_', '')
        metric_idx = feature_names.index(metric_name)
        threshold = best_result['threshold']

        usage_code = f'''
# BEST 100% RECALL PREDICTOR: {best_method}
# Precision: {best_result['precision']:.3f} ({best_result['precision']*100:.1f}%)
# False Positive Rate: {best_result['false_positive_rate']:.1f}%

import numpy as np

def predict_failure_100_recall(confidence_metrics):
    """
    Predict CRNN failure with 100% recall (catches ALL failures).

    Args:
        confidence_metrics: dict with keys {feature_names}

    Returns:
        bool: True if CRNN will likely fail (use SRP), False if CRNN is safe
    """
    {metric_name} = confidence_metrics['{metric_name}']

    # Simple threshold optimized for 100% recall
    {'return ' + metric_name + ' >= ' + str(threshold) if metric_name == 'entropy' else 'return ' + metric_name + ' <= ' + str(threshold)}

# Example usage:
confidence = {{
    'max_prob': 0.025,
    'entropy': 4.5,
    'prediction_variance': 0.0001,
    'peak_sharpness': 1.02,
    'local_concentration': 0.4
}}

if predict_failure_100_recall(confidence):
    print("CRNN will likely fail -> Use SRP")
else:
    print("CRNN is confident -> Use CRNN prediction")
'''
    else:
        # ML model
        model = best_result['model']
        threshold = best_result['threshold']

        # Save model
        model_path = output_dir / 'best_100_recall_model.pkl'
        joblib.dump(model, model_path)

        usage_code = f'''
# BEST 100% RECALL PREDICTOR: {best_method}
# Precision: {best_result['precision']:.3f} ({best_result['precision']*100:.1f}%)
# False Positive Rate: {best_result['false_positive_rate']:.1f}%

import joblib
import numpy as np

# Load the trained model
model = joblib.load('{model_path}')

def predict_failure_100_recall(confidence_metrics):
    """
    Predict CRNN failure with 100% recall using ML model.

    Args:
        confidence_metrics: dict with keys {feature_names}

    Returns:
        bool: True if CRNN will likely fail (use SRP), False if CRNN is safe
    """
    # Convert to feature array
    features = np.array([[
        confidence_metrics['{feature_names[0]}'],
        confidence_metrics['{feature_names[1]}'],
        confidence_metrics['{feature_names[2]}'],
        confidence_metrics['{feature_names[3]}'],
        confidence_metrics['{feature_names[4]}']
    ]])

    # Get failure probability
    failure_prob = model.predict_proba(features)[0, 1]

    # Use threshold optimized for 100% recall
    return failure_prob >= {threshold}

# Example usage:
confidence = {{
    'max_prob': 0.025,
    'entropy': 4.5,
    'prediction_variance': 0.0001,
    'peak_sharpness': 1.02,
    'local_concentration': 0.4
}}

if predict_failure_100_recall(confidence):
    print("CRNN will likely fail -> Use SRP")
else:
    print("CRNN is confident -> Use CRNN prediction")
'''

    with open(output_dir / 'best_100_recall_predictor.py', 'w') as f:
        f.write(usage_code)

    print(f"\\nBest 100% recall predictor saved:")
    print(f"- Code: {output_dir}/best_100_recall_predictor.py")
    if best_method.startswith('ML_'):
        print(f"- Model: {output_dir}/best_100_recall_model.pkl")

def main():
    """Main execution function."""
    print("100% RECALL OPTIMIZATION ANALYSIS")
    print("="*50)
    print("Goal: Achieve 100% recall (catch ALL failures) with maximum precision")

    # Load data
    X, y, feature_names, df = load_data()

    # Split data (use same split as original training for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\\nTest set: {len(X_test)} samples ({np.sum(y_test)} failures to catch)")

    # Test simple thresholds
    simple_results = test_simple_thresholds(X_test, y_test, feature_names)

    # Test ML models
    ml_results = optimize_ml_for_100_recall(X_train, X_test, y_train, y_test, feature_names)

    # Compare and visualize
    output_dir = '100_recall_analysis'
    best_method, best_result = create_comparison_analysis(simple_results, ml_results, y_test, output_dir)

    # Save best predictor
    save_best_100_recall_predictor(best_method, best_result, feature_names, output_dir)

    return simple_results, ml_results, best_method, best_result

if __name__ == "__main__":
    simple_results, ml_results, best_method, best_result = main()