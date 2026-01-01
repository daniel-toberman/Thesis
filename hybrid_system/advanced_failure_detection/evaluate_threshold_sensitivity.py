#!/usr/bin/env python3
"""
Threshold Sensitivity Analysis for OOD Methods

This script evaluates the performance of various Out-of-Distribution (OOD)
methods across a range of 50 different thresholds. It analyzes how 'hybrid_mae'
and 'hybrid_success' metrics change with the 'routing_rate'.

The analysis is performed for multiple microphone configurations, and the results
are categorized based on the number of "changed" microphones (mics with labels >= 9).

Finally, it generates two plots for each method:
1. Routing Rate vs. Hybrid MAE
2. Routing Rate vs. Hybrid Success Rate

Each plot contains four subplots to show the results for all microphone
configurations combined, and for configurations with 2, 3, and 4 changed mics.
A mean performance line is overlaid on each subplot.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Import all router classes from the parent directory
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
from oracle_ood_routing import BestOracleRouter

def load_cached_srp_results(srp_path):
    """Load cached SRP predictions from a pickle file."""
    with open(srp_path, 'rb') as f:
        srp_results_list = pickle.load(f)
    srp_results_df = pd.DataFrame(srp_results_list)
    print(f"  Loaded cached SRP results from {Path(srp_path).name}: {len(srp_results_df)} samples")
    return srp_results_df

def load_crnn_features(crnn_path):
    """Load CRNN features and metadata, returning features as object arrays for flexibility."""
    if not os.path.exists(crnn_path):
        print(f"  ❌ CRNN features file not found: {crnn_path}")
        return None

    if str(crnn_path).endswith('.pkl'):
        with open(crnn_path, 'rb') as f:
            crnn_results_list = pickle.load(f)
        
        features = {
            'abs_errors': np.array([d['crnn_error'] for d in crnn_results_list]),
            'predicted_angles': np.array([d['crnn_pred'] for d in crnn_results_list]),
            'gt_angles': np.array([d['gt_angle'] for d in crnn_results_list]),
            'logits_pre_sig': np.array([d['logits_pre_sig'] for d in crnn_results_list], dtype=object),
            'penultimate_features': np.array([d['penultimate_features'] for d in crnn_results_list], dtype=object),
            'max_prob': np.array([d.get('max_prob', 0.0) for d in crnn_results_list]),
        }
    elif str(crnn_path).endswith('.npz'):
        data = np.load(crnn_path, allow_pickle=True)
        features = {key: data[key] for key in data.files}
    else:
        raise ValueError(f"Unsupported file type: {crnn_path}")
    
    print(f"  Loaded CRNN features from {Path(crnn_path).name}: {len(features['abs_errors'])} samples")
    return features

def count_changed_mics(mic_config_str):
    """Count the number of mics with a label >= 9."""
    mics = mic_config_str.split('_')
    return sum(1 for m_str in mics if m_str.isdigit() and int(m_str) >= 9)

def get_router_and_scores(method_name, features, val_features, srp_results):
    """Initializes router and computes scores for a given method."""
    if method_name == 'best_oracle':
        router = BestOracleRouter()
        scores = router.compute_improvement_scores(features, srp_results)
        return scores

    router_classes = {
        'energy': EnergyOODRouter, 'mc_dropout_entropy': MCDropoutRouter, 'mc_dropout_variance': MCDropoutRouter,
        'knn_k5': lambda: KNNOODRouter(k=5), 'knn_k10': lambda: KNNOODRouter(k=10), 'knn_k20': lambda: KNNOODRouter(k=20),
        'mahalanobis': MahalanobisOODRouter, 'gradnorm': GradNormOODRouter, 'vim': VIMOODRouter, 'she': SHEOODRouter,
        'dice_80': lambda: DICEOODRouter(clip_percentile=80), 'dice_90': lambda: DICEOODRouter(clip_percentile=90),
        'dice_50': lambda: DICEOODRouter(clip_percentile=50), 'max_prob': MaxProbRouter,
    }

    base_method = method_name.split('_T')[0]
    if base_method not in router_classes:
        raise ValueError(f"Unknown base method: {base_method}")

    router = router_classes[base_method]()

    if hasattr(router, 'train'):
        router.train(val_features)

    if base_method == 'energy' and router.temperature is None:
        router.temperature = 1.58

    score_computation_methods = {
        'energy': lambda r, f: r.compute_energy_scores(f['logits_pre_sig']),
        'mc_dropout_entropy': lambda r, f: r.compute_entropy_from_logits(f['logits_pre_sig']),
        'mc_dropout_variance': lambda r, f: r.compute_variance_from_logits(f['logits_pre_sig']),
        'knn_k5': lambda r, f: r.compute_knn_distances(f),
        'knn_k10': lambda r, f: r.compute_knn_distances(f),
        'knn_k20': lambda r, f: r.compute_knn_distances(f),
        'mahalanobis': lambda r, f: r.compute_mahalanobis_distances(f),
        'gradnorm': lambda r, f: r.compute_gradnorm_scores(f),
        'vim': lambda r, f: r.compute_vim_scores(f),
        'she': lambda r, f: r.compute_she_scores(f),
        'dice_80': lambda r, f: r.compute_dice_scores(f),
        'dice_90': lambda r, f: r.compute_dice_scores(f),
        'dice_50': lambda r, f: r.compute_dice_scores(f),
        'llr_gmm5': lambda r, f: r.compute_llr_scores(f),
        'max_prob': lambda r, f: f['max_prob'],
    }

    if base_method not in score_computation_methods:
        raise ValueError(f"Score computation not defined for: {base_method}")
        
    scores = score_computation_methods[base_method](router, features)
    return scores

def evaluate_for_threshold(scores, threshold, ascending, test_features, srp_results):
    """Calculates performance metrics for a single threshold."""
    if threshold == '0%':
        route_decisions = np.zeros(len(test_features['abs_errors']), dtype=bool)
    elif threshold == '100%':
        route_decisions = np.ones(len(test_features['abs_errors']), dtype=bool)
    else:
        avg_scores = scores
        if isinstance(scores, list) or (isinstance(scores, np.ndarray) and scores.dtype == 'object'):
            avg_scores = np.array([s.mean() for s in scores if s is not None and s.size > 0])
        route_decisions = avg_scores > threshold if ascending else avg_scores < threshold
    
    routing_rate = np.mean(route_decisions) * 100
    
    hybrid_preds = np.where(route_decisions, srp_results['srp_pred'].values, test_features['predicted_angles'])
    gt_angles = test_features['gt_angles']
    hybrid_errors = np.minimum(np.abs(hybrid_preds - gt_angles), 360 - np.abs(hybrid_preds - gt_angles))
    
    hybrid_mae = np.mean(hybrid_errors)
    hybrid_success = np.mean(hybrid_errors <= 5.0) * 100
    
    return routing_rate, hybrid_mae, hybrid_success

def generate_and_save_plots(df, output_dir):
    """Generates and saves the performance plots for each method."""
    plot_methods = df['method'].unique()
    
    for method in plot_methods:
        method_df = df[df['method'] == method]
        print(f"  Plotting for method: {method}")
        
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        fig1.suptitle(f'Routing Rate vs. Hybrid MAE for {method}', fontsize=16)
        
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        fig2.suptitle(f'Routing Rate vs. Hybrid Success Rate for {method}', fontsize=16)
        
        categories = [
            ('All Mic Configs', method_df),
            ('2 Mics Changed', method_df[method_df['num_changed_mics'] == 2]),
            ('3 Mics Changed', method_df[method_df['num_changed_mics'] == 3]),
            ('4 Mics Changed', method_df[method_df['num_changed_mics'] == 4]),
        ]
        
        for i, (title, data_category) in enumerate(categories):
            ax1 = axes1.flatten()[i]
            ax2 = axes2.flatten()[i]
            
            if not data_category.empty:
                data = data_category.copy()
                sns.scatterplot(data=data, x='routing_rate', y='hybrid_mae', ax=ax1, alpha=0.3, hue='mic_config', legend=False)
                sns.scatterplot(data=data, x='routing_rate', y='hybrid_success', ax=ax2, alpha=0.3, hue='mic_config', legend=False)
                
                # Use a copy for the intermediate data to avoid the warning
                data_intermediate = data[~data['routing_rate'].isin([0, 100])].copy()
                data_0_100 = data[data['routing_rate'].isin([0, 100])]
                
                if not data_intermediate.empty:
                    bins = np.linspace(0.01, 99.99, 20)
                    labels = (bins[:-1] + bins[1:]) / 2
                    data_intermediate['routing_bin'] = pd.cut(data_intermediate['routing_rate'], bins=bins, labels=labels, include_lowest=True)
                    
                    mean_mae_inter = data_intermediate.groupby('routing_bin', observed=False)['hybrid_mae'].mean().reset_index()
                    mean_success_inter = data_intermediate.groupby('routing_bin', observed=False)['hybrid_success'].mean().reset_index()
                else:
                    mean_mae_inter = pd.DataFrame(columns=['routing_bin', 'hybrid_mae'])
                    mean_success_inter = pd.DataFrame(columns=['routing_bin', 'hybrid_success'])

                mean_0 = data_0_100[data_0_100['routing_rate'] == 0][['hybrid_mae', 'hybrid_success']].mean()
                mean_0['routing_bin'] = 0
                mean_100 = data_0_100[data_0_100['routing_rate'] == 100][['hybrid_mae', 'hybrid_success']].mean()
                mean_100['routing_bin'] = 100

                mean_mae_full = pd.concat([pd.DataFrame([mean_0]), mean_mae_inter, pd.DataFrame([mean_100])], ignore_index=True).sort_values('routing_bin')
                mean_success_full = pd.concat([pd.DataFrame([mean_0]), mean_success_inter, pd.DataFrame([mean_100])], ignore_index=True).sort_values('routing_bin')

                sns.lineplot(data=mean_mae_full, x='routing_bin', y='hybrid_mae', ax=ax1, color='navy', label='Mean MAE', marker='o', errorbar=None)
                sns.lineplot(data=mean_success_full, x='routing_bin', y='hybrid_success', ax=ax2, color='navy', label='Mean Success', marker='o', errorbar=None)

            ax1.set_title(title)
            ax1.set_xlabel('Routing Rate (%)')
            ax1.set_ylabel('Hybrid MAE (°)')
            ax1.grid(True, alpha=0.4)
            
            ax2.set_title(title)
            ax2.set_xlabel('Routing Rate (%)')
            ax2.set_ylabel('Hybrid Success Rate (%)')
            ax2.grid(True, alpha=0.4)

        for fig in [fig1, fig2]:
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
        fig1.savefig(Path(output_dir) / f'{method}_mae_vs_routing.png', dpi=300)
        plt.close(fig1)

        fig2.savefig(Path(output_dir) / f'{method}_success_vs_routing.png', dpi=300)
        plt.close(fig2)

def main():
    """Main execution function."""
    print("="*80)
    print("STARTING THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80)

    output_dir = 'results/threshold_sensitivity'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    val_features_path = "../../crnn features/test_6cm_features.npz"
    val_features = load_crnn_features(val_features_path)

    methods_to_evaluate = [
        'best_oracle', 'energy', 'energy_T1.00', 'vim', 'vim_T0.50', 'she', 'gradnorm',
        'max_prob', 'max_prob_T4.00', 'knn_k5', 'knn_k10', 'knn_k20',
        'mc_dropout_entropy', 'mc_dropout_entropy_T2.00', 
        'mc_dropout_variance', 'mc_dropout_variance_T3.00',
        'dice_80', 'dice_90', 'mahalanobis', 'dice_50'
    ]

    srp_mic_config_files = glob.glob('srp_features_end_result/srp_results_mics_*.pkl')
    all_results = []
    
    for srp_path in srp_mic_config_files:
        mic_config_name_full = Path(srp_path).stem
        mic_config_name = mic_config_name_full.replace('srp_results_mics_', '')
        print(f"\nProcessing mic config: {mic_config_name}")

        crnn_path = f"../../crnn features/crnn_results_mics_{mic_config_name}.pkl"
        
        test_features = load_crnn_features(crnn_path)
        srp_results = load_cached_srp_results(srp_path)
        
        if test_features is None: continue

        num_changed_mics = count_changed_mics(mic_config_name)

        for method in methods_to_evaluate:
            try:
                # Add 0% and 100% routing points for each method and config
                _, crnn_mae, crnn_success = evaluate_for_threshold(None, '0%', True, test_features, srp_results)
                _, srp_mae, srp_success = evaluate_for_threshold(None, '100%', True, test_features, srp_results)
                all_results.append({'method': method, 'mic_config': mic_config_name, 'num_changed_mics': num_changed_mics, 'threshold': np.inf, 'routing_rate': 0.0, 'hybrid_mae': crnn_mae, 'hybrid_success': crnn_success})
                all_results.append({'method': method, 'mic_config': mic_config_name, 'num_changed_mics': num_changed_mics, 'threshold': -np.inf, 'routing_rate': 100.0, 'hybrid_mae': srp_mae, 'hybrid_success': srp_success})

                scores = get_router_and_scores(method, test_features, val_features, srp_results)
                
                if method == 'best_oracle':
                    ascending = True
                else:
                    fail_mask = test_features['abs_errors'] > 5.0
                    avg_scores = scores
                    if isinstance(scores, list) or (isinstance(scores, np.ndarray) and scores.dtype == 'object'):
                        avg_scores = np.array([s.mean() for s in scores if s is not None and s.size > 0])
                    
                    relevant_scores = avg_scores[fail_mask]
                    ascending = np.mean(relevant_scores) > np.mean(avg_scores) if relevant_scores.size > 0 else True

                flat_scores = scores
                if isinstance(scores, list) or (isinstance(scores, np.ndarray) and scores.dtype == 'object'):
                    flat_scores = np.concatenate([s.flatten() for s in scores if s is not None])

                thresholds = np.percentile(flat_scores, np.linspace(1, 99, 50))
                
                for threshold in thresholds:
                    routing_rate, hybrid_mae, hybrid_success = evaluate_for_threshold(
                        scores, threshold, ascending, test_features, srp_results
                    )
                    all_results.append({
                        'method': method,
                        'mic_config': mic_config_name,
                        'num_changed_mics': num_changed_mics,
                        'threshold': threshold,
                        'routing_rate': routing_rate,
                        'hybrid_mae': hybrid_mae,
                        'hybrid_success': hybrid_success,
                    })
            except Exception as e:
                print(f"  ❌ Failed to process method '{method}' for config '{mic_config_name}': {e}")

    results_df = pd.DataFrame(all_results)
    csv_path = Path(output_dir) / 'threshold_sensitivity_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Full results saved to: {csv_path}")

    print("\nGenerating plots...")
    generate_and_save_plots(results_df, output_dir)
    
    print("\n✅ ANALYSIS COMPLETE!")

if __name__ == '__main__':
    main()