#!/usr/bin/python
# -*- coding: utf-8 -*-
# Embedding Quality

"""
This module contains Embedding Quality measures based on coranking matrix.


Author: Alexandre Péré
"""


import numpy as np
import scipy as scp
import scipy.spatial
import sklearn.isotonic
import sklearn.neighbors
import sklearn.metrics
import networkx as nx

def coranking_matrix(X_s, X_l, use_geodesic=False):
    """
    This function allows to construct coranking matrix based on data in state and latent space. Based on implementation
    https://github.com/samueljackson92/coranking .

    Args:
        + X_s: a point cloud in state space
        + X_l: a point cloud in latent space
        + use_geodesic: Whether to use the geodesic distance for state space.
    """

    # We retrieve dimensions of the data
    n, m = X_s.shape
    # We compute distance matrices in both spaces
    if use_geodesic:
        k = 2
        is_connex = False
        while is_connex == False:
            knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k)
            knn.fit(X_s)
            M = knn.kneighbors_graph(X_s, mode='distance')
            graph = nx.from_scipy_sparse_matrix(M)
            is_connex = nx.is_connected(graph)
            k+= 1
        s_distances = nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight')
        s_distances = np.array([np.array(a.items())[:,1] for a in np.array(s_distances.items())[:,1]])
        s_distances = (s_distances + s_distances.T)/2
    else:
        s_distances = scp.spatial.distance.pdist(X_s)
        s_distances = scp.spatial.distance.squareform(s_distances)
    l_distances = scp.spatial.distance.pdist(X_l)
    l_distances = scp.spatial.distance.squareform(l_distances)
    # For each point, we get rank of each point (take care of the weird way .argsort() works)
    s_ranking = s_distances.argsort(axis=1).argsort(axis=1)
    l_ranking = l_distances.argsort(axis=1).argsort(axis=1)
    # We compute the Coranking matrix
    Q, xedges, yedges = np.histogram2d(s_ranking.flatten(),
                                       l_ranking.flatten(),
                                       bins=n)
    # We remove the rankings corresponding to themselves
    Q = Q[1:, 1:]

    return Q

def trustworthiness(Q,k):
    """
    This method allows to compute the Trustworthiness measure, as proposed in
    https://www.researchgate.net/publication/6995963_Local_multidimensional_scaling

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We retrieve the number of points
    n = Q.shape[0]
    # We compute the values
    columns, rows = np.meshgrid(range(n), range(n))
    rank = rows+1
    vals = (rank - k) * Q
    # We compute the mask
    mask = np.zeros([n,n])
    mask[k:,:k] = 1.
    # We compute the normalization constant
    norm = n*k*(2.*n-3.*k-1.) if k<(n/2) else n*(n-k)*(n-k-1.)
    # We finally compute the measures
    measure = 1. - 2./(float(norm)) * (vals*mask).sum()

    return measure


def continuity(Q,k):
    """
    This method allows to compute the Continuity measure, as proposed in
    https://www.researchgate.net/publication/6995963_Local_multidimensional_scaling

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We retrieve the number of points
    n = Q.shape[0]
    # We compute the values
    columns, rows = np.meshgrid(range(n), range(n))
    rank = columns+1
    vals = (rank - k) * Q
    # We compute the mask
    mask = np.zeros([n,n])
    mask[:k,k:] = 1.
    # We compute the normalization constant
    norm = n*k*(2.*n-3.*k-1.) if k<(n/2) else n*(n-k)*(n-k-1.)
    # We finally compute the measures
    measure = 1. - 2./(float(norm)) * (vals*mask).sum()

    return measure

def mrre_trustworthiness(Q,k):
    """
    This method allows to compute the mean relative rank error Trustworthiness measure, as proposed in
    https://www.researchgate.net/publication/221165927_Rank-based_quality_assessment_of_nonlinear_dimensionality_reduction

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We retrieve the number of points
    n = Q.shape[0]
    # We compute the values
    columns, rows = np.meshgrid(range(n), range(n))
    rank_c = columns+1
    rank_r = rows+1
    vals = np.abs( (rank_r-rank_c)/rank_c*Q )
    # We compute the mask
    mask = np.zeros([n,n])
    mask[:k,:k] = 1.
    mask[k:,:k] = 1.
    # We compute the normalization constant
    norm = n * np.abs(n-2.*np.arange(1,k+1) / np.arange(1,k+1)).sum()
    # We finally compute the measures
    measure = (vals*mask).sum() / float(norm)

    return measure

def mrre_continuity(Q,k):
    """
    This method allows to compute the mean relative rank error Continuity measure, as proposed in
    https://www.researchgate.net/publication/221165927_Rank-based_quality_assessment_of_nonlinear_dimensionality_reduction

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We retrieve the number of points
    n = Q.shape[0]
    # We compute the values
    columns, rows = np.meshgrid(range(n), range(n))
    rank_c = columns+1
    rank_r = rows+1
    vals = np.abs( (rank_r-rank_c)/rank_c*Q )
    # We compute the mask
    mask = np.zeros([n,n])
    mask[:k,:k] = 1.
    mask[:k,k:] = 1.
    # We compute the normalization constant
    norm = n * np.abs(n-2.*np.arange(1,k+1) / np.arange(1,k+1)).sum()
    # We finally compute the measures
    measure = (vals*mask).sum() / float(norm)

    return measure


def lcmc(Q,k):
    """
    This method allows to compute the local continuity meta criterion measure, as proposed in
    https://www.researchgate.net/publication/227369127_Local_Multidimensional_Scaling_for_Nonlinear_Dimension_Reduction_Graph_Drawing_and_Proximity_Analysis

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We retrieve the number of points
    n = Q.shape[0]
    # We compute the values
    vals = Q
    # We compute the mask
    mask = np.zeros([n,n])
    mask[:k,:k] = 1.
    # We finally compute the measures
    measure = (k/(1.-n)) + (1./(n*k)) *  (vals*mask).sum()

    return measure

def intrusion(Q,k):
    """
    This method allows to compute the fraction of intrusion.

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We retrieve the number of points
    n = Q.shape[0]
    # We compute the values
    vals = Q
    # We compute the mask
    mask = np.zeros([n,n])
    mask[:k,:k] = np.triu(np.ones([k,k]))
    # We compute the normalization constant
    norm = k * (n+1.)
    # We finally compute the measures
    measure = (vals*mask).sum() / float(norm)

    return measure

def extrusion(Q,k):
    """
    This method allows to compute the fraction of extrusion.

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We retrieve the number of points
    n = Q.shape[0]
    # We compute the values
    vals = Q
    # We compute the mask
    mask = np.zeros([n,n])
    mask[:k,:k] = np.tril(np.ones([k,k]))
    # We compute the normalization constant
    norm = k * (n+1.)
    # We finally compute the measures
    measure = (vals*mask).sum() / float(norm)

    return measure

def protrusion(Q,k):
    """
    This method allows to compute the fraction of protrusion (samerank).

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We retrieve the number of points
    n = Q.shape[0]
    # We compute the values
    vals = Q
    # We compute the mask
    mask = np.zeros([n,n])
    mask[:k,:k] = np.eye(k)
    # We compute the normalization constant
    norm = k * (n+1.)
    # We finally compute the measures
    measure = (vals*mask).sum() / float(norm)

    return measure


def qnx(Q,k):
    """
    This method allows to compute the coranking matrix quality criterion.

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We retrieve the number of points
    n = Q.shape[0]
    # We compute the values
    vals = Q
    # We compute the mask
    mask = np.zeros([n,n])
    mask[:k,:k] = 1.
    # We compute the normalization constant
    norm = k * (n+1.)
    # We finally compute the measures
    measure = (vals*mask).sum() / float(norm)

    return measure

def qtc(Q,k):
    """
    This method allows to compute the trustworthiness and continuity criterion.

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: The coranking matrix
        + k: The neighbourhood size
    """

    # We compute the measure
    measure = trustworthiness(Q,k) + continuity(Q,k)

    return measure

def qmrre(Q,k):
    """
    This method allows to compute the mrre quality criterion.

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: the coranking matrix
        + k: the neighbourhood size
    """

    # We compute the measures
    measure = 2 - mrre_trustworthiness(Q,k) - mrre_continuity(Q,k)

    return measure

def bnx(Q,k):
    """
    This method allows to compute the coranking matrix behavior (intrusive if > 0, extrusive otherwise)

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: the coranking matrix
        + k: the neighbourhood size
    """

    # We compute the measure
    measure = intrusion(Q,k)-extrusion(Q,k)

    return measure

def btc(Q,k):
    """
    This method allows to compute the T&C behavior.

    Implementation based on https://github.com/gdkrmr/coRanking .

    Args:
        + Q: the coranking matrix
        + k: the neighbourhood size
    """

    # We compute the measure
    measure = trustworthiness(Q,k)-continuity(Q,k)

    return measure


def stress(X_s, X_l, use_geodesic=True):
    """
    This method allows to compute multiple stress functions:
        + Kruskal stress https://www.researchgate.net/publication/24061688_Nonmetric_multidimensional_scaling_A_numerical_method
        + S stress http://gifi.stat.ucla.edu/janspubs/1977/articles/takane_young_deleeuw_A_77.pdf
        + Sammon stress http://ieeexplore.ieee.org/document/1671271/?reload=true
        + Quadratic Loss

    Args:
        + X_s: the point cloud in state space
        + X_l: the point cloud in latent space
        + use_geodesic: Whether to use geodesic distance for state space
    """

    # We retrieve dimensions of the data
    n, m = X_s.shape
    # We compute distance matrices in both spaces
    if use_geodesic:
        k = 2
        is_connex = False
        while is_connex == False:
            knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k)
            knn.fit(X_s)
            M = knn.kneighbors_graph(X_s, mode='distance')
            graph = nx.from_scipy_sparse_matrix(M)
            is_connex = nx.is_connected(graph)
            k+= 1
        s_uni_distances = nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight')
        s_all_distances = np.array([np.array(a.items())[:,1] for a in np.array(s_uni_distances.items())[:,1]])
        s_all_distances = (s_all_distances+s_all_distances.T)/2
        s_uni_distances = scp.spatial.distance.squareform(s_all_distances)
        s_all_distances = s_all_distances.ravel()
    else:
        s_uni_distances = scp.spatial.distance.pdist(X_s)
        s_all_distances = scp.spatial.squareform(s_uni_distances).ravel()
    l_uni_distances = scp.spatial.distance.pdist(X_l)
    l_all_distances = scp.spatial.distance.squareform(l_uni_distances).ravel()
    # We set up the measure dict
    measures = dict()
    # 1. Quadratic Loss
    measures['quadratic_loss'] = np.square(s_uni_distances - l_uni_distances).sum()
    # 2. Sammon stress
    measures['sammon_stress'] = (1/s_uni_distances.sum())*(np.square(s_uni_distances-l_uni_distances)/s_uni_distances).sum()
    # 3. S stress
    measures['s_stress'] = np.sqrt((1/n)*(np.square((np.square(s_uni_distances)-np.square(l_uni_distances)).sum())/np.power(s_uni_distances,4))).sum()
    # 4. Kruskal stress
    # We reorder the distances under the order of distances in latent space
    s_all_distances = s_all_distances[l_all_distances.argsort()]
    l_all_distances = l_all_distances[l_all_distances.argsort()]
    # We perform the isotonic regression
    iso = sklearn.isotonic.IsotonicRegression()
    s_iso_distances = iso.fit_transform(s_all_distances, l_all_distances)
    # We compute the kruskal stress
    measures['kruskal_stress'] = np.sqrt(np.square(s_iso_distances-l_all_distances).sum() / np.square(l_all_distances).sum())

    return measures

def residual_variance(X_s, X_l, use_geodesic=True):
    """
    This function allows to compute the residual variance as proposed in
    https://www.researchgate.net/publication/12204039_A_Global_Geometric_Framework_for_Nonlinear_Dimensionality_Reduction

    Args:
        + X_s: the point cloud in state space
        + X_l: the point cloud in latent space
        + use_geodesic: Whether to use geodesic distance for state space
    """

    # We retrieve dimensions of the data
    n, m = X_s.shape
    # We compute distance matrices in both spaces
    if use_geodesic:
        k = 2
        is_connex = False
        while is_connex == False:
            knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k)
            knn.fit(X_s)
            M = knn.kneighbors_graph(X_s, mode='distance')
            graph = nx.from_scipy_sparse_matrix(M)
            is_connex = nx.is_connected(graph)
            k+= 1
        s_uni_distances = nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight')
        s_all_distances = np.array([np.array(a.items())[:,1] for a in np.array(s_uni_distances.items())[:,1]])
        s_all_distances = (s_all_distances+s_all_distances.T)/2
        s_all_distances = s_all_distances.ravel()
    else:
        s_uni_distances = scp.spatial.distance.pdist(X_s)
        s_all_distances = scp.spatial.distance.squareform(s_uni_distances).ravel()
    l_uni_distances = scp.spatial.distance.pdist(X_l)
    l_all_distances = scp.spatial.distance.squareform(l_uni_distances).ravel()
    # We compute the residual variance
    measure = sklearn.metrics.r2_score(s_all_distances, l_all_distances)

    return measure

def metric_distorsion(X_s, X_l, use_geodesic=True):
    """
    This function allows to compute the empirical metric distorsion.

    Args:
        + X_s: the point cloud in state space
        + X_l: the point cloud in latent space
        + use_geodesic: whether to use the geodesic distance for state space
    """

    # We retrieve dimensions of the data
    n, m = X_s.shape
    # We compute distance matrices in both spaces
    if use_geodesic:
        k = 2
        is_connex = False
        while is_connex == False:
            knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k)
            knn.fit(X_s)
            M = knn.kneighbors_graph(X_s, mode='distance')
            graph = nx.from_scipy_sparse_matrix(M)
            is_connex = nx.is_connected(graph)
            k+= 1
        s_uni_distances = nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight')
        s_all_distances = np.array([np.array(a.items())[:,1] for a in np.array(s_uni_distances.items())[:,1]])
        s_all_distances = (s_all_distances + s_all_distances.T)/2
        s_uni_distances = scipy.spatial.distance.squareform(s_all_distances)
    else:
        s_uni_distances = scp.spatial.distance.pdist(X_s)
    l_uni_distances = scp.spatial.distance.pdist(X_l)
    # We compute the contraction
    contraction = s_uni_distances/(l_uni_distances+1e-6)
    # We compute the expansion
    expansion = l_uni_distances/(s_uni_distances+1e-6)
    # We compute the distorsion
    distorsion = contraction[np.where(l_uni_distances != 0.)].max()*expansion[np.where(s_uni_distances!= 0.)].max()

    return distorsion
