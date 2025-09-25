#!/usr/bin/env python3
"""
Train failure predictor using CRNN confidence metrics.
Test multiple approaches: SVM, Random Forest, XGBoost, Neural Network.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
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
    print(f"Features: {confidence_features}")
    print(f"Class distribution: {np.bincount(y)} (Success: {np.sum(y==0)}, Failure: {np.sum(y==1)})")
    print(f"Class balance: {np.sum(y==1)/len(y)*100:.1f}% failures")

    return X, y, confidence_features, df

def create_models():
    """Create different model pipelines to test."""
    models = {
        'SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ]),

        'SVM_Linear': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', probability=True, random_state=42))
        ]),

        'RandomForest': Pipeline([
            ('rf', RandomForestClassifier(random_state=42, n_estimators=100))
        ]),

        'XGBoost': Pipeline([
            ('gb', GradientBoostingClassifier(random_state=42, n_estimators=100))
        ]),

        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]),

        'NeuralNet': Pipeline([
            ('scaler', StandardScaler()),
            ('nn', MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500))
        ])
    }

    return models

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for best models."""

    # SVM RBF tuning
    svm_params = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }

    # Random Forest tuning
    rf_params = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [10, 20, None],
        'rf__min_samples_split': [2, 5, 10]
    }

    # XGBoost tuning
    gb_params = {
        'gb__n_estimators': [50, 100, 200],
        'gb__learning_rate': [0.01, 0.1, 0.2],
        'gb__max_depth': [3, 5, 7]
    }

    models_to_tune = {
        'SVM_RBF': (Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=42))]), svm_params),
        'RandomForest': (Pipeline([('rf', RandomForestClassifier(random_state=42))]), rf_params),
        'XGBoost': (Pipeline([('gb', GradientBoostingClassifier(random_state=42))]), gb_params)
    }

    tuned_models = {}
    print("\\nPerforming hyperparameter tuning...")

    for name, (model, params) in models_to_tune.items():
        print(f"Tuning {name}...")

        grid_search = GridSearchCV(
            model, params, cv=5, scoring='f1',
            n_jobs=-1, verbose=0
        )

        grid_search.fit(X_train, y_train)
        tuned_models[f'{name}_Tuned'] = grid_search.best_estimator_

        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV F1 score: {grid_search.best_score_:.3f}")

    return tuned_models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate all models using cross-validation and test set."""
    results = {}

    print("\\nEvaluating models...")
    print("="*80)
    print(f"{'Model':<20} {'CV F1':<8} {'Test F1':<8} {'Test Prec':<10} {'Test Rec':<9} {'Test AUC':<9}")
    print("-"*80)

    for name, model in models.items():
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        cv_f1 = cv_scores.mean()

        # Fit and predict on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        test_f1 = f1_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_pred_proba)

        results[name] = {
            'cv_f1': cv_f1,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model
        }

        print(f"{name:<20} {cv_f1:.3f}    {test_f1:.3f}    {test_precision:.3f}     {test_recall:.3f}     {test_auc:.3f}")

    return results

def create_visualizations(results, X_test, y_test, feature_names):
    """Create visualization plots for model performance."""
    output_dir = Path('failure_predictor_results')
    output_dir.mkdir(exist_ok=True)

    # 1. Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    models = list(results.keys())
    cv_f1_scores = [results[m]['cv_f1'] for m in models]
    test_f1_scores = [results[m]['test_f1'] for m in models]
    test_precision_scores = [results[m]['test_precision'] for m in models]
    test_recall_scores = [results[m]['test_recall'] for m in models]

    # F1 scores
    axes[0, 0].bar(models, cv_f1_scores, alpha=0.7, label='CV F1')
    axes[0, 0].bar(models, test_f1_scores, alpha=0.7, label='Test F1')
    axes[0, 0].set_title('F1 Score Comparison')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Precision vs Recall
    axes[0, 1].scatter(test_precision_scores, test_recall_scores, s=100, alpha=0.7)
    for i, model in enumerate(models):
        axes[0, 1].annotate(model, (test_precision_scores[i], test_recall_scores[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_xlabel('Precision')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Precision vs Recall')
    axes[0, 1].grid(True, alpha=0.3)

    # ROC curves
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC={result['test_auc']:.3f})")
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Feature importance for Random Forest
    if 'RandomForest' in results:
        rf_model = results['RandomForest']['model']
        feature_importance = rf_model.named_steps['rf'].feature_importances_
        axes[1, 1].bar(feature_names, feature_importance)
        axes[1, 1].set_title('Feature Importance (Random Forest)')
        axes[1, 1].set_ylabel('Importance')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confusion matrices for best models
    best_models = sorted(results.keys(), key=lambda x: results[x]['test_f1'], reverse=True)[:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, model_name in enumerate(best_models):
        cm = confusion_matrix(y_test, results[model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{model_name}\\nF1: {results[model_name]["test_f1"]:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\\nVisualization saved to: {output_dir.absolute()}")

def save_best_model(results, feature_names):
    """Save the best performing model."""
    # Find best model by F1 score
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_f1'])
    best_model = results[best_model_name]['model']
    best_performance = results[best_model_name]

    print(f"\\n" + "="*60)
    print("BEST MODEL SUMMARY")
    print(f"="*60)
    print(f"Model: {best_model_name}")
    print(f"CV F1 Score: {best_performance['cv_f1']:.3f}")
    print(f"Test F1 Score: {best_performance['test_f1']:.3f}")
    print(f"Test Precision: {best_performance['test_precision']:.3f}")
    print(f"Test Recall: {best_performance['test_recall']:.3f}")
    print(f"Test AUC: {best_performance['test_auc']:.3f}")
    print()

    # Save model using joblib
    import joblib
    model_path = 'failure_predictor_results/best_failure_predictor.pkl'
    joblib.dump({
        'model': best_model,
        'feature_names': feature_names,
        'performance': best_performance,
        'model_name': best_model_name
    }, model_path)

    print(f"Best model saved to: {model_path}")

    # Create a simple prediction function example
    prediction_code = f'''
# Example usage of the trained failure predictor:
import joblib
import numpy as np

# Load the model
model_data = joblib.load('{model_path}')
model = model_data['model']
feature_names = model_data['feature_names']

# Example prediction (replace with actual confidence metrics)
def predict_failure(max_prob, entropy, prediction_variance, peak_sharpness, local_concentration):
    features = np.array([[max_prob, entropy, prediction_variance, peak_sharpness, local_concentration]])
    failure_prob = model.predict_proba(features)[0, 1]
    failure_prediction = model.predict(features)[0]

    return {{
        'will_fail': bool(failure_prediction),
        'failure_probability': failure_prob,
        'confidence_threshold': 0.5  # Adjust based on precision/recall trade-off
    }}

# Usage example:
result = predict_failure(0.02, 4.8, 0.0001, 1.02, 0.35)
if result['will_fail']:
    print(f"CRNN likely to fail (prob: {{result['failure_probability']:.3f}}) -> Use SRP")
else:
    print(f"CRNN likely to succeed (prob: {{1-result['failure_probability']:.3f}}) -> Use CRNN")
    '''

    with open('failure_predictor_results/usage_example.py', 'w') as f:
        f.write(prediction_code)

    return best_model_name, best_performance

def main():
    """Main execution function."""
    print("CRNN Failure Predictor Training")
    print("="*50)

    # Load and prepare data
    X, y, feature_names, df = load_and_prepare_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\\nTrain set: {len(X_train)} samples ({np.sum(y_train)} failures)")
    print(f"Test set: {len(X_test)} samples ({np.sum(y_test)} failures)")

    # Create base models
    models = create_models()

    # Hyperparameter tuning for promising models
    tuned_models = hyperparameter_tuning(X_train, y_train)

    # Combine all models
    all_models = {**models, **tuned_models}

    # Evaluate models
    results = evaluate_models(all_models, X_train, X_test, y_train, y_test)

    # Create visualizations
    create_visualizations(results, X_test, y_test, feature_names)

    # Save best model
    best_model_name, best_performance = save_best_model(results, feature_names)

    return results, best_model_name, best_performance

if __name__ == "__main__":
    results, best_model, performance = main()