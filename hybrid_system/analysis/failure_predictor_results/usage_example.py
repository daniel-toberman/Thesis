
# Example usage of the trained failure predictor:
import joblib
import numpy as np

# Load the model
model_data = joblib.load('failure_predictor_results/best_failure_predictor.pkl')
model = model_data['model']
feature_names = model_data['feature_names']

# Example prediction (replace with actual confidence metrics)
def predict_failure(max_prob, entropy, prediction_variance, peak_sharpness, local_concentration):
    features = np.array([[max_prob, entropy, prediction_variance, peak_sharpness, local_concentration]])
    failure_prob = model.predict_proba(features)[0, 1]
    failure_prediction = model.predict(features)[0]

    return {
        'will_fail': bool(failure_prediction),
        'failure_probability': failure_prob,
        'confidence_threshold': 0.5  # Adjust based on precision/recall trade-off
    }

# Usage example:
result = predict_failure(0.02, 4.8, 0.0001, 1.02, 0.35)
if result['will_fail']:
    print(f"CRNN likely to fail (prob: {result['failure_probability']:.3f}) -> Use SRP")
else:
    print(f"CRNN likely to succeed (prob: {1-result['failure_probability']:.3f}) -> Use CRNN")
    