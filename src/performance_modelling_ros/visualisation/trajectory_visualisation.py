#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from os import path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def save_trajectories_plot(run_output_folder):
    """
    Creates a figure with the trajectories slam and ground truth
    """
    base_link_poses_path = path.join(run_output_folder, "base_link_poses")
    ground_truth_poses_path = path.join(run_output_folder, "ground_truth_poses")
    figure_output_path = path.join(run_output_folder, "trajectories.svg")

    with open(base_link_poses_path, 'r') as base_link_poses:
        x = list()
        y = list()
        for line in base_link_poses:
            words = line.split(" ")
            if len(words) != 9:
                print("save_trajectory_plot: unexpected file format for {path}".format(path=base_link_poses_path))
                return
            x.append(float(words[5]))
            y.append(float(words[6]))

    with open(ground_truth_poses_path, 'r') as ground_truth_poses:
        x_gt = list()
        y_gt = list()
        for line in ground_truth_poses:
            words = line.split(" ")
            if len(words) != 4:
                print("save_trajectory_plot: unexpected file format for {path}".format(path=ground_truth_poses_path))
                return
            x_gt.append(float(words[1]))
            y_gt.append(float(words[2]))

    fig, ax = plt.subplots()
    ax.cla()
    ax.plot(x, y, 'r', linewidth=0.25)
    ax.plot(x_gt, y_gt, 'b', linewidth=0.25)
    fig.savefig(figure_output_path)
    plt.close(fig)
