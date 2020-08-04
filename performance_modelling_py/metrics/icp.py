#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


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


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    scores = list()

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
        scores.append(np.mean(distances))

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, scores

# import numpy as np
# from scipy.spatial import KDTree
#
#
# class Align2D:
#
#     # params:
#     #   source_points: numpy array containing points to align to the target set
#     #                  points should be homogeneous, with one point per row
#     #   target_points: numpy array containing points to which the source points
#     #                  are to be aligned, points should be homogeneous with one
#     #                  point per row
#     #   initial_T:     initial estimate of the transform between target and source
#     def __init__(self, source_points, target_points, initial_t):
#         self.source = source_points
#         self.target = target_points
#         self.init_T = initial_t
#         self.target_tree = KDTree(target_points[:, :2])
#         self.transform = self.align_icp(20, 1.0e-4)
#
#     # uses the iterative closest point algorithm to find the
#     # transformation between the source and target point clouds
#     # that minimizes the sum of squared errors between nearest
#     # neighbors in the two point clouds
#     # params:
#     #   max_iter: int, max number of iterations
#     #   min_delta_err: float, minimum change in alignment error
#     def align_icp(self, max_iter, min_delta_err):
#
#         mean_sq_error = 1.0e6  # initialize error as large number
#         delta_err = 1.0e6  # change in error (used in stopping condition)
#         T = self.init_T
#         num_iter = 0  # number of iterations
#         tf_source = self.source
#
#         while delta_err > min_delta_err and num_iter < max_iter:
#
#             # find correspondences via nearest-neighbor search
#             matched_trg_pts, matched_src_pts, indices = self.find_correspondences(tf_source)
#
#             # find alignment between source and corresponding target points via SVD
#             # note: svd step doesn't use homogeneous points
#             new_T = self.align_svd(matched_src_pts, matched_trg_pts)
#
#             # update transformation between point sets
#             T = np.dot(T, new_T)
#
#             # apply transformation to the source points
#             tf_source = np.dot(self.source, T.T)
#
#             # find mean squared error between transformed source points and target points
#             new_err = 0
#             for i in range(len(indices)):
#                 if indices[i] != -1:
#                     diff = tf_source[i, :2] - self.target[indices[i], :2]
#                     new_err += np.dot(diff, diff.T)
#
#             new_err /= float(len(matched_trg_pts))
#
#             # update error and calculate delta error
#             delta_err = abs(mean_sq_error - new_err)
#             mean_sq_error = new_err
#
#             print("delta_error:", delta_err)
#
#             num_iter += 1
#
#         return T
#
#     # finds nearest neighbors in the target point for all points
#     # in the set of source points
#     # params:
#     #   src_pts: array of source points for which we will find neighbors
#     #            points are assumed to be homogeneous
#     # returns:
#     #   array of nearest target points to the source points (not homogeneous)
#     def find_correspondences(self, src_pts):
#
#         # get distances to nearest neighbors and indices of nearest neighbors
#         matched_src_pts = src_pts[:, :2]
#         dist, indices = self.target_tree.query(matched_src_pts)
#
#         # remove multiple associations from index list
#         # only retain closest associations
#         unique = False
#         while not unique:
#             unique = True
#             for i in range(len(indices)):
#                 if indices[i] == -1:
#                     continue
#                 for j in range(i + 1, len(indices)):
#                     if indices[i] == indices[j]:
#                         if dist[i] < dist[j]:
#                             indices[j] = -1
#                         else:
#                             indices[i] = -1
#                             break
#         # build array of nearest neighbor target points
#         # and remove unmatched source points
#         point_list = []
#         src_idx = 0
#         for idx in indices:
#             if idx != -1:
#                 point_list.append(self.target[idx, :])
#                 src_idx += 1
#             else:
#                 matched_src_pts = np.delete(matched_src_pts, src_idx, axis=0)
#
#         matched_pts = np.array(point_list)
#
#         return matched_pts[:, :2], matched_src_pts, indices
#
#     # uses singular value decomposition to find the
#     # transformation from the target to the source point cloud
#     # assumes source and target point clouds are ordered such that
#     # corresponding points are at the same indices in each array
#     #
#     # params:
#     #   source: numpy array representing source pointcloud
#     #   target: numpy array representing target pointcloud
#     # returns:
#     #   T: transformation between the two point clouds
#     def align_svd(self, source, target):
#
#         # first find the centroids of both point clouds
#         src_centroid = self.get_centroid(source)
#         trg_centroid = self.get_centroid(target)
#
#         # get the point clouds in reference to their centroids
#         source_centered = source - src_centroid
#         target_centered = target - trg_centroid
#
#         # get cross covariance matrix M
#         M = np.dot(target_centered.T, source_centered)
#
#         # get singular value decomposition of the cross covariance matrix
#         U, W, V_t = np.linalg.svd(M)
#
#         # get rotation between the two point clouds
#         R = np.dot(U, V_t)
#
#         # get the translation (simply the difference between the point cloud centroids)
#         t = np.expand_dims(trg_centroid, 0).T - np.dot(R, np.expand_dims(src_centroid, 0).T)
#
#         # assemble translation and rotation into a transformation matrix
#         T = np.identity(3)
#         T[:2, 2] = np.squeeze(t)
#         T[:2, :2] = R
#
#         return T
#
#     @staticmethod
#     def get_centroid(points):
#         point_sum = np.sum(points, axis=0)
#         return point_sum / float(len(points))
