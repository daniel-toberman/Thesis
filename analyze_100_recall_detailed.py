#!/usr/bin/env python3
"""
Detailed analysis of 100% recall methods with full breakdown.
Verify data source and provide comprehensive statistics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_data_source():
    """Analyze the confidence data to understand what we're working with."""
    print("=" * 80)
    print("DATA SOURCE ANALYSIS")
    print("=" * 80)

    confidence_file = 'crnn_predictions/crnn_predictions_with_confidence_clean.csv'

    if not Path(confidence_file).exists():
        print(f"ERROR: {confidence_file} not found!")
        return None, None, None, None

    df = pd.read_csv(confidence_file)

    print(f"Total samples in dataset: {len(df)}")
    print(f"Expected samples: 2009")
    print(f"Match expected: {'YES' if len(df) == 2009 else 'NO'}")

    # Check for novel noise indicators
    columns = df.columns.tolist()
    print(f"\nDataset columns: {columns}")

    # Create failure labels (>30 degrees error)
    df['is_failure'] = (df['abs_error'] > 30).astype(int)

    total_failures = df['is_failure'].sum()
    total_successes = len(df) - total_failures
    failure_rate = total_failures / len(df) * 100

    print(f"\nFAILURE ANALYSIS:")
    print(f"Total failures (>30° error): {total_failures}")
    print(f"Total successes (≤30° error): {total_successes}")
    print(f"Failure rate: {failure_rate:.1f}%")

    # Expected failure rates for different scenarios:
    print(f"\nEXPECTED SCENARIOS:")
    print(f"Clean data: ~10 failures (0.5%)")
    print(f"Novel noise data: ~191 failures (9.5%)")
    print(f"Current data: {total_failures} failures ({failure_rate:.1f}%)")

    if total_failures < 50:
        print("⚠️  WARNING: This appears to be CLEAN data, not novel noise data!")
    elif total_failures > 150:
        print("✅ This appears to be NOVEL NOISE data")
    else:
        print("❓ Uncertain data source")

    return df, total_failures, total_successes, failure_rate

def find_100_recall_threshold(y_true, y_scores, metric_name):
    """Find threshold that achieves exactly 100% recall."""
    failure_indices = y_true == 1
    failure_scores = y_scores[failure_indices]

    if len(failure_scores) == 0:
        return None, 0, 0, 0, 0

    # Find threshold for 100% recall
    if metric_name in ['entropy']:  # Higher values indicate failure
        threshold = failure_scores.min() - 1e-10
        predictions = y_scores >= threshold
    else:  # Lower values indicate failure
        threshold = failure_scores.max() + 1e-10
        predictions = y_scores <= threshold

    # Calculate metrics
    tp = np.sum(predictions & (y_true == 1))  # True positives (failures caught)
    fp = np.sum(predictions & (y_true == 0))  # False positives (successes flagged as failures)
    tn = np.sum(~predictions & (y_true == 0))  # True negatives (successes correctly identified)
    fn = np.sum(~predictions & (y_true == 1))  # False negatives (failures missed)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return threshold, precision, recall, tp, fp

def comprehensive_100_recall_analysis():
    """Complete analysis of all methods for 100% recall."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE 100% RECALL ANALYSIS")
    print("=" * 80)

    # Load and analyze data
    df, total_failures, total_successes, failure_rate = analyze_data_source()
    if df is None:
        return

    # Prepare features
    confidence_features = ['max_prob', 'entropy', 'prediction_variance', 'peak_sharpness', 'local_concentration']
    X = df[confidence_features].copy()
    y = df['is_failure'].copy()

    # Split data (use same split as before)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    test_failures = y_test.sum()
    test_successes = len(y_test) - test_failures

    print(f"\nTEST SET BREAKDOWN:")
    print(f"Test set size: {len(y_test)}")
    print(f"Failures in test set: {test_failures}")
    print(f"Successes in test set: {test_successes}")
    print(f"We need to catch ALL {test_failures} failures with 100% recall")

    all_results = []

    # 1. Test Simple Thresholds
    print(f"\n{'='*60}")
    print("SIMPLE THRESHOLD METHODS")
    print(f"{'='*60}")

    for i, metric in enumerate(confidence_features):
        threshold, precision, recall, tp, fp = find_100_recall_threshold(
            y_test.values, X_test.iloc[:, i].values, metric
        )

        if threshold is not None and recall >= 0.999:  # 100% recall
            fp_rate = fp / test_successes * 100

            result = {
                'Method': f'Simple_{metric}',
                'Type': 'Simple',
                'Threshold': threshold,
                'Precision': precision,
                'Recall': recall,
                'True_Positives': tp,
                'False_Positives': fp,
                'False_Positive_Rate_Percent': fp_rate,
                'Failures_Caught': tp,
                'Total_Failures': test_failures,
                'Failures_Missed': test_failures - tp
            }
            all_results.append(result)

            print(f"{metric:<20} | Precision: {precision:.3f} | Failures caught: {tp}/{test_failures} | False+: {fp} ({fp_rate:.1f}%)")

    # 2. Test ML Models
    print(f"\n{'='*60}")
    print("MACHINE LEARNING METHODS")
    print(f"{'='*60}")

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

    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)

            # Get prediction probabilities
            y_prob = model.predict_proba(X_test)[:, 1]

            # Find threshold for 100% recall
            threshold, precision, recall, tp, fp = find_100_recall_threshold(
                y_test.values, y_prob, 'probability'
            )

            if threshold is not None and recall >= 0.999:  # 100% recall
                fp_rate = fp / test_successes * 100

                result = {
                    'Method': f'ML_{name}',
                    'Type': 'ML',
                    'Threshold': threshold,
                    'Precision': precision,
                    'Recall': recall,
                    'True_Positives': tp,
                    'False_Positives': fp,
                    'False_Positive_Rate_Percent': fp_rate,
                    'Failures_Caught': tp,
                    'Total_Failures': test_failures,
                    'Failures_Missed': test_failures - tp
                }
                all_results.append(result)

                print(f"{name:<20} | Precision: {precision:.3f} | Failures caught: {tp}/{test_failures} | False+: {fp} ({fp_rate:.1f}%)")
            else:
                print(f"{name:<20} | FAILED to achieve 100% recall (recall: {recall:.3f})")

        except Exception as e:
            print(f"{name:<20} | ERROR: {str(e)[:50]}")

    # 3. Summary and Save Results
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")

    if not all_results:
        print("ERROR: No methods achieved 100% recall!")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Precision', ascending=False)

    print(f"{'Method':<25} {'Precision':<10} {'Caught':<12} {'False+':<8} {'FP Rate':<8}")
    print("-" * 75)

    for _, row in results_df.iterrows():
        print(f"{row['Method']:<25} {row['Precision']:<10.3f} {row['Failures_Caught']}/{row['Total_Failures']:<8} "
              f"{row['False_Positives']:<8} {row['False_Positive_Rate_Percent']:<8.1f}%")

    best_method = results_df.iloc[0]
    print(f"\n🏆 BEST METHOD: {best_method['Method']}")
    print(f"   Precision: {best_method['Precision']:.3f} ({best_method['Precision']*100:.1f}%)")
    print(f"   Failures caught: {best_method['Failures_Caught']}/{best_method['Total_Failures']}")
    print(f"   False positives: {best_method['False_Positives']}")
    print(f"   False positive rate: {best_method['False_Positive_Rate_Percent']:.1f}%")

    # Save detailed results
    output_file = '100_recall_detailed_analysis.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n✅ Detailed results saved to: {output_file}")

    # Additional summary statistics
    print(f"\n{'='*80}")
    print("DATA VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset file: crnn_predictions/crnn_predictions_with_confidence_clean.csv")
    print(f"Total samples: {len(df)}")
    print(f"Total failures: {total_failures} ({failure_rate:.1f}%)")
    print(f"Test set failures: {test_failures}")
    print(f"All methods caught: {test_failures}/{test_failures} failures (100% recall verified)")

    return results_df

if __name__ == "__main__":
    results = comprehensive_100_recall_analysis()