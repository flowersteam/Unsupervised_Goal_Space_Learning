#!/usr/bin/python
# -*- coding: utf-8 -*-
# Measures

"""
This module contains measure methods to assess performance of learned manifolds.
"""

import numpy as np
import scipy.spatial as spatial
from skl_groups.divergences import KNNDivergenceEstimator
from skl_groups.features import Features
import embqual


def distribution_divergence(X_s, X_l, k=10):
    """
    This function computes l2 and js divergences from samples of two distributions.
    The implementation use `skl-groups`, which implements non-parametric estimation
    of divergences.

    Args:
        + X_s: a numpy array containing point cloud in state space
        + X_e: a numpy array containing point cloud in latent space
    """

    # We discard cases with too large dimensions
    if X_s.shape[1] > 50:
        return {'l2_divergence': -1., 'js_divergence': -1.}

    # We instantiate the divergence object
    div = KNNDivergenceEstimator(div_funcs=['l2','js'], Ks=[k], n_jobs=4, clamp=True)

    # We turn both data to float32
    X_s = X_s.astype(np.float32)
    X_l = X_l.astype(np.float32)

    # We generate Features
    f_s = Features(X_s, n_pts=[X_s.shape[0]])
    f_l = Features(X_l, n_pts=[X_l.shape[0]])

    # We create the knn graph
    div.fit(X=f_s)

    # We compute the divergences
    l2, js = div.transform(X=f_l).squeeze()

    # We construct the returned dictionnary
    output = {'l2_divergence':l2, 'js_divergence':js}

    return output

def embedding_local_quality(X_s, X_l, k_range=[7]):
    """
    This function computes various embedding local quality measures based on coranking matrix.

    Args:
        + X_s: a numpy array containing point cloud in state space
        + X_l: a numpy array containing point cloud in latent space
    """

    # We compute the coranking matrix
    Q = embqual.coranking_matrix(X_s,X_l)

    # We construct the returned dictionnary
    output = {'trustworthiness':[embqual.trustworthiness(Q,k) for k in k_range],
              'continuity':[embqual.continuity(Q,k) for k in k_range],
              'mrre_trustworthiness': [embqual.mrre_trustworthiness(Q,k) for k in k_range],
              'mrre_continuity': [embqual.mrre_continuity(Q,k) for k in k_range],
              'lcmc': [embqual.lcmc(Q,k) for k in k_range],
              'intrusion': [embqual.intrusion(Q,k) for k in k_range],
              'extrusion': [embqual.extrusion(Q,k) for k in k_range],
              'protrusion': [embqual.protrusion(Q,k) for k in k_range],
              'qnx':[embqual.qnx(Q,k) for k in k_range],
              'qtc': [embqual.qtc(Q,k) for k in k_range],
              'qmrre': [embqual.qmrre(Q,k) for k in k_range],
              'bnx': [embqual.bnx(Q,k) for k in k_range],
              'btc': [embqual.btc(Q,k) for k in k_range]}

    return output

def embedding_global_quality(X_s, X_l):
    """
    This function computes various embedding global quality measures.

    Args:
        + X_s: the point cloud in state space
        + X_l: the point cloud in latent space
    """

    # We initialize the output dictionnary
    output = dict()
    # We add stress
    output.update(embqual.stress(X_s, X_l))
    # We add residual variance
    output['residual_variance'] = embqual.residual_variance(X_s, X_l)
    # We add empirical distorsion
    output['empirical_distorsion'] = embqual.metric_distorsion(X_s, X_l)

    return output

def exploration(explorer, bins=10):
    """
    This function computes the exploration measure, i.e. the number of cells reached in the state space.

    Args:
        + explorer: the explorer of the environment
        + bins: the number of bins per dimension
    """
    # We retrieve useful informations
    states_list = np.array(explorer.get_reached_states())
    mins, maxs = explorer.get_states_limits()
    # We retrieve the dimension of the state space
    n = len(mins)
    # We return 0 if no iterations were performed
    if len(states_list) == 0:
        return 0
    # We compute the
    else:
        assert len(states_list[0]) == n
        hist, _ = np.histogramdd(states_list, bins=[bins]*n, range=np.array([mins, maxs]).T)
    return float(hist[hist>0].size) / float(hist.size)

def model_mse(explorer, nb_samples=10000):
    """
    This method computes the mse of the inverse model

    Args:
        + explorer: the explorer contraining models
    """

    # We retrieve useful informations
    mins, maxs = explorer.get_states_limits()
    n=len(mins)
    # We sample random goals in complete state space
    state_goals = np.random.uniform(mins, maxs,[nb_samples, n])
    state_achie = np.zeros([nb_samples, n])
    # We compute the achieved states
    for i in range(nb_samples):
        state_achie[i] = explorer.exploit(state_goals[i])
    # We compute the average error
    err = np.square(np.linalg.norm(state_goals-state_achie , axis=1, ord=2)).mean()

    return err
