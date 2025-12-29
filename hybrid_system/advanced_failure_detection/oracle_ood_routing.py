#!/usr/bin/env python3
"""
Best Oracle OOD Router for Failure Detection

This router represents the theoretical best performance for a given routing rate.
It calculates the potential error improvement for every sample and uses this as a
score, allowing a thresholding mechanism to select the most impactful cases to route.
"""

import numpy as np

class BestOracleRouter:
    """
    An oracle that routes cases based on the maximum potential error improvement.
    The "score" is defined as: crnn_error - srp_error.
    A higher score means routing to SRP is more beneficial.
    """
    def __init__(self):
        """Initializes the router."""
        pass

    def compute_improvement_scores(self, test_features, srp_results):
        """
        Computes the potential error improvement for each sample.

        Args:
            test_features (dict): Dictionary containing CRNN results,
                                  including 'abs_errors' (crnn_error).
            srp_results (pd.DataFrame): DataFrame containing SRP results,
                                        including 'srp_error'.

        Returns:
            np.ndarray: An array of improvement scores.
        """
        crnn_errors = test_features['abs_errors']
        srp_errors = srp_results['srp_error'].values
        
        if len(crnn_errors) != len(srp_errors):
            raise ValueError("CRNN and SRP results must have the same length.")
            
        improvement_scores = crnn_errors - srp_errors
        return improvement_scores
