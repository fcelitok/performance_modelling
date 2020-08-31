#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(a, b):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      transform: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      rotation: mxm rotation matrix
      translation: mx1 translation vector
    """

    assert a.shape == b.shape

    # get number of dimensions
    m = a.shape[1]

    # translate points to their centroids
    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)
    aa = a - centroid_a
    bb = b - centroid_b

    # rotation matrix
    h = np.dot(aa.T, bb)
    u, s, v_t = np.linalg.svd(h)
    rotation = np.dot(v_t.T, u.T)

    # special reflection case
    if np.linalg.det(rotation) < 0:
        v_t[m - 1, :] *= -1
        rotation = np.dot(v_t.T, u.T)

    # translation
    translation = centroid_b.T - np.dot(rotation, centroid_a.T)

    # homogeneous transformation
    transform = np.identity(m + 1)
    transform[:m, :m] = rotation
    transform[:m, m] = translation

    return transform, rotation, translation


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def iterative_closest_point(a, b, max_iterations=1, tolerance=0.001, init_pose=None):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        transform: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    # assert A.shape == B.shape

    # get number of dimensions
    m = a.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, a.shape[0]))
    dst = np.ones((m + 1, b.shape[0]))
    src[:m, :] = np.copy(a.T)
    dst[:m, :] = np.copy(b.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    scores = list()

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
        # scores.append(np.mean((distances+1)**2))
        scores.append(np.mean(distances))

        # compute the transformation between the current source and nearest destination points
        transform, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(transform, src)

        # check error
        # mean_error = np.mean((distances+1)**2)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    transform, _, _ = best_fit_transform(a, src[:m, :].T)
    return transform, scores, src[:m, :].T, i + 1
