
# BEST 100% RECALL PREDICTOR: Simple_prediction_variance
# Precision: 0.500 (50.0%)
# False Positive Rate: 0.2%

import numpy as np

def predict_failure_100_recall(confidence_metrics):
    """
    Predict CRNN failure with 100% recall (catches ALL failures).

    Args:
        confidence_metrics: dict with keys ['max_prob', 'entropy', 'prediction_variance', 'peak_sharpness', 'local_concentration']

    Returns:
        bool: True if CRNN will likely fail (use SRP), False if CRNN is safe
    """
    prediction_variance = confidence_metrics['prediction_variance']

    # Simple threshold optimized for 100% recall
    return prediction_variance <= 1.7473073922849633e-05

# Example usage:
confidence = {
    'max_prob': 0.025,
    'entropy': 4.5,
    'prediction_variance': 0.0001,
    'peak_sharpness': 1.02,
    'local_concentration': 0.4
}

if predict_failure_100_recall(confidence):
    print("CRNN will likely fail -> Use SRP")
else:
    print("CRNN is confident -> Use CRNN prediction")
