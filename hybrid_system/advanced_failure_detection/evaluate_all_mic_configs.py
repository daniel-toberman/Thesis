#!/usr/bin/env python3
"""
Evaluate All OOD Methods with Optimal F1 Thresholds for All Mic Configs

Uses optimal thresholds from distribution analysis to compute hybrid metrics.
SRP results are cached, so this runs quickly.
"""

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import glob

# Import all router classes
from energy_ood_routing import EnergyOODRouter
from mc_dropout_routing import MCDropoutRouter
from knn_ood_routing import KNNOODRouter
from mahalanobis_ood_routing import MahalanobisOODRouter
from gradnorm_ood_routing import GradNormOODRouter
from vim_ood_routing import VIMOODRouter
from she_ood_routing import SHEOODRouter
from dice_ood_routing import DICEOODRouter
from llr_ood_routing import LLROODRouter
from max_prob_routing import MaxProbRouter


def load_cached_srp_results(srp_path):
    """Load cached SRP predictions."""
    with open(srp_path, 'rb') as f:
        srp_results_list = pickle.load(f)
    
    # Convert list of dicts to DataFrame
    srp_results_df = pd.DataFrame(srp_results_list)
    print(f"Loaded cached SRP results from {srp_path}: {len(srp_results_df)} samples")
    return srp_results_df


def load_crnn_features(crnn_path):


    """Load CRNN features and metadata from a pickle or npz file."""


    if crnn_path.endswith('.pkl'):


        with open(crnn_path, 'rb') as f:


            crnn_results_list = pickle.load(f)





        # Handle potentially inhomogeneous shapes for 'logits_pre_sig' and 'penultimate_features'


        logits_pre_sig_list = [d['logits_pre_sig'] for d in crnn_results_list]


        penultimate_features_list = [d['penultimate_features'] for d in crnn_results_list]





        try:


            logits_pre_sig_array = np.stack(logits_pre_sig_list)


        except ValueError:


            print(f"Warning: logits_pre_sig has inhomogeneous shapes in {crnn_path}. Loading as object array.")


            logits_pre_sig_array = np.array(logits_pre_sig_list, dtype=object)





        try:


            penultimate_features_array = np.stack(penultimate_features_list)


        except ValueError:


            print(f"Warning: penultimate_features has inhomogeneous shapes in {crnn_path}. Loading as object array.")


            penultimate_features_array = np.array(penultimate_features_list, dtype=object)





        features = {


            'abs_errors': np.array([d['crnn_error'] for d in crnn_results_list]),


            'predicted_angles': np.array([d['crnn_pred'] for d in crnn_results_list]),


            'gt_angles': np.array([d['gt_angle'] for d in crnn_results_list]),


            'logits_pre_sig': logits_pre_sig_array,


            'penultimate_features': penultimate_features_array,


            'max_prob': np.array([d['max_prob'] for d in crnn_results_list]),


        }


        print(f"Loaded CRNN features from {crnn_path}: {len(features['abs_errors'])} samples")


        return features


    elif crnn_path.endswith('.npz'):


        data = np.load(crnn_path, allow_pickle=True)


        features = {key: data[key] for key in data.files}


        # The npz file has slightly different key names


        if 'abs_errors' not in features:


            features['abs_errors'] = features['crnn_error']


        if 'predicted_angles' not in features:


            features['predicted_angles'] = features['crnn_pred']





        print(f"Loaded CRNN features from {crnn_path}: {len(features['abs_errors'])} samples")


        return features


    else:


        raise ValueError(f"Unsupported file type: {crnn_path}")


    





def load_optimal_thresholds():


    """Load optimal F1 thresholds from distribution analysis."""


    # Load original methods


    df1 = pd.read_csv('results/ood_distributions/distribution_analysis_summary.csv')





    # Load temperature-scaled methods


    df2 = pd.read_csv('results/ood_distributions/temperature_scaling_summary.csv')





    # Combine


    df = pd.concat([df1, df2], ignore_index=True)





    thresholds = {}


    for _, row in df.iterrows():


        method = row['method']


        optimal_thresh = row['optimal_threshold']


        optimal_f1 = row['optimal_f1']


        roc_auc = row['roc_auc']


        thresholds[method] = {


            'threshold': optimal_thresh,


            'f1': optimal_f1,


            'roc_auc': roc_auc


        }





    print(f"Loaded optimal thresholds for {len(thresholds)} methods")


    return thresholds








def compute_method_scores_with_temp(method_name, features, val_features, temperature=None):
    """
    Compute OOD scores for a method, with optional temperature scaling.

    Args:
        method_name: Base method name (e.g., 'energy', 'max_prob')
        features: Test features
        val_features: Validation features (for training if needed)
        temperature: Temperature scaling factor (optional)
    """
    from scipy.special import softmax
    from scipy.stats import entropy

    # Apply temperature scaling to logits if temperature is provided
    if temperature is not None and temperature != 1.0:
        features = features.copy()
        features['logits_pre_sig'] = features['logits_pre_sig'] / temperature

        val_features = val_features.copy()
        val_features['logits_pre_sig'] = val_features['logits_pre_sig'] / temperature

    # Compute scores based on method
    if method_name == 'energy':
        router = EnergyOODRouter()
        router.temperature = 1.0 if temperature is None else temperature
        scores = router.compute_energy_scores(features['logits_pre_sig'])

    elif method_name == 'max_prob':
        router = MaxProbRouter()
        scores = features['max_prob']

    elif method_name == 'mc_dropout_entropy':
        router = MCDropoutRouter()
        scores = router.compute_entropy_from_logits(features['logits_pre_sig'])

    elif method_name == 'mc_dropout_variance':
        router = MCDropoutRouter()
        scores = router.compute_variance_from_logits(features['logits_pre_sig'])

    elif method_name == 'vim':
        router = VIMOODRouter()
        router.train(val_features)
        scores = router.compute_vim_scores(features)

    elif method_name == 'she':
        router = SHEOODRouter()
        router.train(val_features)
        scores = router.compute_she_scores(features)

    elif method_name == 'gradnorm':
        router = GradNormOODRouter()
        router.train(val_features)
        scores = router.compute_gradnorm_scores(features)

    elif method_name.startswith('knn_k'):
        k = int(method_name.split('_k')[1])
        router = KNNOODRouter(k=k)
        router.train(val_features)
        scores = router.compute_knn_distances(features)

    elif method_name == 'mahalanobis':
        router = MahalanobisOODRouter()
        router.train(val_features)
        scores = router.compute_mahalanobis_distances(features)

    elif method_name.startswith('dice_'):
        sparsity = int(method_name.split('_')[1])
        router = DICEOODRouter(clip_percentile=sparsity)
        router.train(val_features)
        scores = router.compute_dice_scores(features)

    elif method_name.startswith('llr_gmm'):
        n_components = int(method_name.split('gmm')[1])
        router = LLROODRouter(n_components=n_components)
        router.train(val_features)
        scores = router.compute_llr_scores(features)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    return scores








def evaluate_hybrid_with_threshold(method_display_name, base_method, temperature, threshold,


                                   test_features, val_features, srp_results):


    """


    Evaluate hybrid performance with given threshold.





    Returns:


        dict with routing stats and hybrid metrics


    """


    # Compute scores


    scores = compute_method_scores_with_temp(base_method, test_features, val_features, temperature)





    # Determine routing direction (higher score = route or lower score = route)


    # Auto-detect based on score distribution vs errors


    fail_mask = test_features['abs_errors'] > 5.0


    ascending = scores[fail_mask].mean() > scores[~fail_mask].mean()





    # Apply threshold to get routing decisions


    if ascending:


        route_decisions = scores > threshold


    else:


        route_decisions = scores < threshold





    # Routing statistics


    n_routed = route_decisions.sum()


    routing_rate = n_routed / len(route_decisions) * 100





    # Ground truth: should route if error > 5°


    should_route = test_features['abs_errors'] > 5.0





    # Precision, Recall, F1


    precision = precision_score(should_route, route_decisions, zero_division=0)


    recall = recall_score(should_route, route_decisions, zero_division=0)


    f1 = f1_score(should_route, route_decisions, zero_division=0)





    # False positive rate


    n_success = (~should_route).sum()


    fp = (route_decisions & ~should_route).sum()


    fp_rate = fp / n_success if n_success > 0 else 0





    # Compute hybrid predictions


    crnn_preds = test_features['predicted_angles']


    srp_preds_array = srp_results['srp_pred'].values





    # Hybrid: use SRP where routed, CRNN otherwise


    hybrid_preds = np.where(route_decisions, srp_preds_array, crnn_preds)





    # Compute hybrid metrics


    ground_truth = test_features['gt_angles']


    hybrid_errors = np.abs(hybrid_preds - ground_truth)





    # Handle wraparound at 360/0 degrees


    hybrid_errors = np.minimum(hybrid_errors, 360 - hybrid_errors)





    hybrid_mae = hybrid_errors.mean()


    hybrid_median = np.median(hybrid_errors)


    hybrid_success = (hybrid_errors <= 5.0).sum() / len(hybrid_errors) * 100





    # Baseline CRNN metrics


    crnn_mae = test_features['abs_errors'].mean()


    crnn_success = (test_features['abs_errors'] <= 5.0).sum() / len(test_features['abs_errors']) * 100





    # Improvements


    delta_mae = hybrid_mae - crnn_mae


    delta_success = hybrid_success - crnn_success





    return {


        'method': method_display_name,


        'base_method': base_method,


        'temperature': temperature,


        'threshold': threshold,


        'routing_rate': routing_rate,


        'n_routed': int(n_routed),


        'precision': precision,


        'recall': recall,


        'f1': f1,


        'fp_rate': fp_rate,


        'hybrid_mae': hybrid_mae,


        'hybrid_median': hybrid_median,


        'hybrid_success': hybrid_success,


        'delta_mae': delta_mae,


        'delta_success': delta_success,


        'crnn_mae': crnn_mae,


        'crnn_success': crnn_success


    }








def evaluate_single_config(srp_results, test_features, val_features, optimal_thresholds, mic_config_name):


    """Evaluates all methods for a single microphone configuration."""


    


    method_configs = [


        # Original methods


        ('energy', 'energy', None),


        ('vim', 'vim', None),


        ('she', 'she', None),


        ('gradnorm', 'gradnorm', None),


        ('max_prob', 'max_prob', None),


        ('knn_k5', 'knn_k5', None),


        ('knn_k10', 'knn_k10', None),


        ('knn_k20', 'knn_k20', None),


        ('mc_dropout_entropy', 'mc_dropout_entropy', None),


        ('mc_dropout_variance', 'mc_dropout_variance', None),


        ('mahalanobis', 'mahalanobis', None),


        ('dice_80', 'dice_80', None),


        ('dice_90', 'dice_90', None),


        # ('llr_gmm5', 'llr_gmm5', None),





        # Temperature-scaled methods


        ('energy_T1.00', 'energy', 1.0),


        ('max_prob_T4.00', 'max_prob', 4.0),


        ('mc_dropout_entropy_T2.00', 'mc_dropout_entropy', 2.0),


        ('mc_dropout_variance_T3.00', 'mc_dropout_variance', 3.0),


        ('vim_T0.50', 'vim', 0.5),


    ]





    results = []





    for i, (method_key, base_method, temperature) in enumerate(method_configs, 1):


        if method_key not in optimal_thresholds:


            continue





        threshold_info = optimal_thresholds[method_key]


        threshold = threshold_info['threshold']





        try:


            result = evaluate_hybrid_with_threshold(


                method_key, base_method, temperature, threshold,


                test_features, val_features, srp_results


            )


            result['mic_config'] = mic_config_name


            results.append(result)





        except Exception as e:


            print(f"  ❌ Error evaluating {method_key} for {mic_config_name}: {e}")


            import traceback


            traceback.print_exc()





    return results








def main():


    """Main execution."""


    print("="*100)


    print("EVALUATING ALL METHODS WITH OPTIMAL F1 THRESHOLDS FOR ALL MIC CONFIGS")


    print("="*100)





    # Load data that is constant for all configs


    print("\nLoading constant data...")


    val_features_path = Path("../../crnn features/test_6cm_features.npz")


    val_features = load_crnn_features(str(val_features_path))


    print(f"Loaded validation features: {len(val_features['abs_errors'])} samples")


    optimal_thresholds = load_optimal_thresholds()


    


    # Find all mic config pkl files


    srp_mic_config_files = glob.glob('features/srp_results_mics_*.pkl')


    print(f"Found {len(srp_mic_config_files)} microphone configurations to evaluate.")





    all_results = []





    for srp_path in srp_mic_config_files:


        mic_config_name = Path(srp_path).stem.replace('srp_results_mics_', '')


        crnn_path = f"../../crnn features/crnn_results_mics_{mic_config_name}.pkl"


        


        print(f"\n{'='*100}")


        print(f"Evaluating mic configuration: {mic_config_name}")


        print(f"{'='*100}")


        


        srp_results = load_cached_srp_results(srp_path)


        test_features = load_crnn_features(crnn_path)


        


        config_results = evaluate_single_config(srp_results, test_features, val_features, optimal_thresholds, mic_config_name)


        all_results.extend(config_results)








    # Save results


    print(f"\n{'='*100}")


    print("SAVING ALL RESULTS")


    print(f"{'='*100}")





    df = pd.DataFrame(all_results)





    # Sort by mic_config and then by hybrid MAE


    df = df.sort_values(['mic_config', 'hybrid_mae'])





    # Save CSV


    output_dir = Path('results/all_mic_configs')


    output_dir.mkdir(parents=True, exist_ok=True)





    csv_path = output_dir / 'all_mic_configs_optimal_thresholds.csv'


    df.to_csv(csv_path, index=False)


    print(f"\n✅ Saved: {csv_path}")





    print(f"\n{'='*100}")


    print("✅ EVALUATION COMPLETE!")


    print(f"{'='*100}")


    print(f"Total microphone configurations evaluated: {len(srp_mic_config_files)}")


    print(f"Total results saved: {len(df)}")


    print(f"Results directory: {output_dir}")








if __name__ == '__main__':


    main()

