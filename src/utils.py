#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains some utilities methods.
"""


import numpy as np
import scipy.spatial as ss
import matplotlib.path




def labels_to_categoricals(y):
    """
    This method allows to turn labels vectors of shape [n,1], into binary categorical vectors of shape [n, m].
    :param y: a numpy labels vector
    :return: a numpy categoricals vector
    """

    assert isinstance(y, np.ndarray)
    assert y.shape[1] == 1

    nb_classes = y.max()
    nb_samples = y.shape[0]
    yout = np.zeros([nb_samples, nb_classes+1], dtype=np.int8)
    for index in range(0, nb_samples):
        yout[index][y[index]] = 1

    return yout

def kldiv(s_q, s_p, k=3):
    """
    D_{kl}(q|p) from samples using k-nn. Based on paper http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.422.5121.
    Implementation based on https://github.com/gregversteeg/NPEET
    """

    # We retrieve dimensions and samples nb.
    d = s_q.shape[1]
    n = s_q.shape[0]
    m = s_p.shape[0]

    # We instantiate KD-Tree for quick nn search
    tree_q = ss.cKDTree(s_q)
    tree_p = ss.cKDTree(s_p)

    # We compute k-th nearest neighbours distance
    dist_q = np.array([tree_q.query(point, k+1., p=2)[0][k] for point in s_q])
    dist_p = np.array([tree_p.query(point, k, p=2)[0][k-1] for point in s_q])

    # We compute the KL-Div estimation
    kl = d*np.mean(np.log(dist_p)) - d*np.mean(np.log(dist_q)) + np.log(m) - np.log((n-1))

    return kl

def manifold_2_space_coordinates(X,order):
    """
    This function allows to transform manifold coordinates to space coordinates. In particular, it encodes cyclic
    in two sin cos dims.

    Args:
        + X: the point cloud in manifold coordinates of size (n,m)
        + order: an array of size (m) containing 1 if dimension is axial, and 2 if dimension is cyclic
    """

    # We turn order into np array
    order = np.array(order)
    # We instantiate the output array
    X_o = np.zeros([X.shape[0], order.sum()])
    # We loop through dims and extend cyclic dims.
    for i in range(order.size):
        if order[i] == 1:
            X_o[:,order[:i+1].sum()-1] = X[:,i]
        if order[i] == 2:
            X_o[:,order[:i+1].sum()-2] = np.sin(X[:,i]*2*np.pi)/np.pi
            X_o[:,order[:i+1].sum()-1] = np.cos(X[:,i]*2*np.pi)/np.pi

    return X_o

def in_convex_hull(p, X):
    """
    This method allows to check whether a point p lies inside of the convex hull of a point cloud X. From
    https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl

    Args:
        + p: the point to consider
        + X: the point cloud to compute the convex hull
    """

    # We compute the convex hull of the points
    hull = ss.Delaunay(X)

    # We check whether the point lies inside the path
    result = hull.find_simplex(p) >= 0

    return result

def uniform_in_hull(nb_samples, X):
    """
    This method allows to sample uniformely in the convex hull of a point cloud X, using rejection sampling.

    Args:
        + nb_samples: the number of samples
        + X: the point cloud to compute the convex hull
    """

    # We compute the convex hull of the points
    hull = ss.Delaunay(X)

    # We retrieve the max and min for uniform sampling
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    # We sample the points
    output = list()
    while len(output)<nb_samples:
            p = np.random.uniform(low=mins, high=maxs, size=X.shape[1])
            if hull.find_simplex(p)>=0:
                output.append(p)

    return np.array(output)
