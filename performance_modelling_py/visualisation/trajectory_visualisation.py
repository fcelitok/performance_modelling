#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
from os import path

import matplotlib as mpl
import pandas as pd

mpl.use('Agg')
import matplotlib.pyplot as plt


def save_trajectories_plot(visualisation_output_folder, estimated_poses_path, estimated_correction_poses_path, ground_truth_poses_path):
    """
    Creates a figure with the trajectories slam and ground truth
    """

    estimated_poses_df = pd.read_csv(estimated_poses_path)
    estimated_correction_poses_df = pd.read_csv(estimated_correction_poses_path)
    ground_truth_poses_df = pd.read_csv(ground_truth_poses_path)

    fig, ax = plt.subplots()
    ax.cla()

    ax.plot(estimated_poses_df['x'], estimated_poses_df['y'], 'red', linewidth=0.25, label='odometry')
    ax.scatter(estimated_correction_poses_df['x'], estimated_correction_poses_df['y'], s=10, c='black', marker='x', linewidth=0.25, label='corrections')
    ax.plot(ground_truth_poses_df['x'], ground_truth_poses_df['y'], 'blue', linewidth=0.25, label='ground truth')
    ax.legend(fontsize='x-small')

    if not path.exists(visualisation_output_folder):
        os.makedirs(visualisation_output_folder)

    figure_output_path = path.join(visualisation_output_folder, "trajectories.svg")
    fig.savefig(figure_output_path)
    plt.close(fig)


def multivariate_gaussian(pos, mu, sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    import numpy as np
    n = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    denom = np.sqrt((2*np.pi)**n * sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, sigma_inv, pos-mu)

    return np.exp(-fac / 2) / denom


def save_trajectories_with_covariance_plot(visualisation_output_folder, estimated_poses_path, estimated_correction_poses_path, ground_truth_poses_path):
    """
    Creates a figure with the trajectories slam and ground truth
    """
    import numpy as np
    from matplotlib import cm
    import matplotlib.cm

    estimated_poses_df = pd.read_csv(estimated_poses_path)
    estimated_correction_poses_df = pd.read_csv(estimated_correction_poses_path)
    ground_truth_poses_df = pd.read_csv(ground_truth_poses_path)

    # Our 2-dimensional distribution will be over variables X and Y
    res = 0.01
    x = np.arange(estimated_poses_df['x'].min()-1, estimated_poses_df['x'].max()+1, res)
    y = np.arange(estimated_poses_df['y'].min()-1, estimated_poses_df['y'].max()+1, res)
    x, y = np.meshgrid(x, y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    # The distribution on the variables X, Y packed into pos.
    z = None
    for _, row in estimated_correction_poses_df.iterrows():
        mu = np.array([row['x'], row['y']])
        sigma = np.array([[row['cov_x_x'], row['cov_x_y']],
                          [row['cov_x_y'], row['cov_y_y']]])
        # sigma = np.array([[.005, .0], [0.,  .005]])
        if z is None:
            z = multivariate_gaussian(pos, mu=mu, sigma=sigma)
        else:
            z = np.maximum(z, multivariate_gaussian(pos, mu=mu, sigma=sigma))

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    ax = fig.gca()
    contour = ax.contourf(x, y, z, cmap=cm.viridis)
    fig.colorbar(contour)
    ax.plot(estimated_poses_df['x'], estimated_poses_df['y'], 'red', linewidth=0.25, label='odometry')
    ax.scatter(estimated_correction_poses_df['x'], estimated_correction_poses_df['y'], s=10, c='black', marker='x', linewidth=0.25, label='corrections')
    ax.plot(ground_truth_poses_df['x'], ground_truth_poses_df['y'], 'blue', linewidth=0.25, label='ground truth')
    ax.legend(fontsize='x-small')

    if not path.exists(visualisation_output_folder):
        os.makedirs(visualisation_output_folder)

    figure_output_path = path.join(visualisation_output_folder, "trajectories_with_covariance.svg")
    fig.savefig(figure_output_path)
    plt.close(fig)
