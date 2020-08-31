#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import math
import scipy
import time

import numpy as np
import pandas as pd
import os
import yaml
from os import path

from performance_modelling_py.environment.ground_truth_map import GroundTruthMap
from scipy.stats import t
import scipy.spatial
import subprocess

from performance_modelling_py.utils import print_info, print_error


def metric_evaluator(exec_path, poses_path, relations_path, weights, log_path, errors_path, unsorted_errors_path=None):
    with open(log_path, 'w') as stdout_log_file:
        if unsorted_errors_path is None:
            p = subprocess.Popen([exec_path, "-s", poses_path, "-r", relations_path, "-w", weights, "-e", errors_path], stdout=stdout_log_file)
        else:
            p = subprocess.Popen([exec_path, "-s", poses_path, "-r", relations_path, "-w", weights, "-e", errors_path, "-eu", unsorted_errors_path], stdout=stdout_log_file)
        p.wait()


def compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df, ground_truth_interpolation_tolerance=0.1):
    estimated_poses_df['t_datetime'] = pd.to_datetime(estimated_poses_df['t'], unit='s')
    ground_truth_poses_df['t_datetime'] = pd.to_datetime(ground_truth_poses_df['t'], unit='s')

    # interpolate ground_truth around estimate times
    tolerance = pd.Timedelta('{}s'.format(ground_truth_interpolation_tolerance))
    forward_matches = pd.merge_asof(
        left=estimated_poses_df[['t_datetime', 't', 'x', 'y', 'theta']],
        right=ground_truth_poses_df[['t_datetime', 't', 'x', 'y', 'theta']],
        on='t_datetime',
        direction='forward',
        tolerance=tolerance,
        suffixes=('_estimate', '_ground_truth'))
    backward_matches = pd.merge_asof(
        left=estimated_poses_df[['t_datetime', 't', 'x', 'y', 'theta']],
        right=ground_truth_poses_df[['t_datetime', 't', 'x', 'y', 'theta']],
        on='t_datetime',
        direction='backward',
        tolerance=tolerance,
        suffixes=('_estimate', '_ground_truth'))
    forward_backward_matches = pd.merge(
        left=backward_matches,
        right=forward_matches,
        on='t_estimate')

    interpolated_ground_truth_df = pd.DataFrame(columns=['t', 'x_est', 'y_est', 'theta_est', 'x_gt', 'y_gt', 'theta_gt'])
    for index, row in forward_backward_matches.iterrows():
        t_gt_1, t_gt_2 = row['t_ground_truth_x'], row['t_ground_truth_y']
        t_int = row['t_estimate']

        # if the estimate time is too far from a ground truth time (before or after), do not use this estimate data point
        if pd.isnull(t_gt_1) or pd.isnull(t_gt_2):
            continue

        x_est = row['x_estimate_x']
        x_gt_1, x_gt_2 = row['x_ground_truth_x'], row['x_ground_truth_y']
        x_int = np.interp(t_int, [t_gt_1, t_gt_2], [x_gt_1, x_gt_2])

        y_est = row['y_estimate_x']
        y_gt_1, y_gt_2 = row['y_ground_truth_x'], row['y_ground_truth_y']
        y_int = np.interp(t_int, [t_gt_1, t_gt_2], [y_gt_1, y_gt_2])

        theta_est = row['theta_estimate_x']
        theta_gt_1, theta_gt_2 = row['theta_ground_truth_x'], row['theta_ground_truth_y']
        theta_int = np.interp(t_int, [t_gt_1, t_gt_2], [theta_gt_1, theta_gt_2])

        interpolated_ground_truth_df = interpolated_ground_truth_df.append({
            't': t_int,
            'x_est': x_est,
            'y_est': y_est,
            'theta_est': theta_est,
            'x_gt': x_int,
            'y_gt': y_int,
            'theta_gt': theta_int,
        }, ignore_index=True)

    return interpolated_ground_truth_df


def relative_localization_error_metrics(log_output_folder, estimated_poses_file_path, ground_truth_poses_file_path, alpha=0.99, max_error=0.02, compute_sequential_relations=False):
    """
    Generates the ordered and the random relations files and computes the metrics
    """

    if not path.exists(log_output_folder):
        os.makedirs(log_output_folder)

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("compute_relative_localization_error: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(ground_truth_poses_file_path):
        print_error("compute_relative_localization_error: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return

    relative_errors_dict = dict()

    # find the metricEvaluator executable
    metric_evaluator_exec_path = path.join(path.dirname(path.abspath(__file__)), "metricEvaluator", "metricEvaluator")

    estimated_poses_df = pd.read_csv(estimated_poses_file_path)
    if len(estimated_poses_df.index) < 2:
        print_error("not enough estimated poses data points")
        return

    ground_truth_poses_df = pd.read_csv(ground_truth_poses_file_path)
    interpolated_ground_truth_df = compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df)

    # if no matching ground truth data points are found, the metrics con not be computed
    if len(interpolated_ground_truth_df.index) < 2:
        print_error("no matching ground truth data points were found")
        return

    # convert estimated_poses_file to the CARMEN format
    estimated_poses_carmen_file_path = path.join(log_output_folder, "estimated_poses_carmen_format")
    with open(estimated_poses_carmen_file_path, "w") as estimated_poses_carmen_file:
        for index, row in estimated_poses_df.iterrows():
            estimated_poses_carmen_file.write("FLASER 0 0.0 0.0 0.0 {x} {y} {theta} {t}\n".format(x=row['x'], y=row['y'], theta=row['theta'], t=row['t']))

    # random relations
    relations_re_file_path = path.join(log_output_folder, "re_relations")
    with open(relations_re_file_path, "w") as relations_file_re:

        n_samples = 500
        for _ in range(n_samples):
            two_random_poses = interpolated_ground_truth_df[['t', 'x_gt', 'y_gt', 'theta_gt']].sample(n=2)
            t_1, x_1, y_1, theta_1 = two_random_poses.iloc[0]
            t_2, x_2, y_2, theta_2 = two_random_poses.iloc[1]

            # reorder data so that t_1 is before t_2
            if t_1 > t_2:
                # swap 1 and 2
                t_1, x_1, y_1, theta_1, t_2, x_2, y_2, theta_2 = t_2, x_2, y_2, theta_2, t_1, x_1, y_1, theta_1

            rel = get_matrix_diff((x_1, y_1, theta_1), (x_2, y_2, theta_2))

            x_relative = rel[0, 3]
            y_relative = rel[1, 3]
            theta_relative = math.atan2(rel[1, 0], rel[0, 0])

            relations_file_re.write("{t_1} {t_2} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(t_1=t_1, t_2=t_2, x=x_relative, y=y_relative, theta=theta_relative))

    # Run the metric evaluator on this relations file, read the sample standard deviation and exploit it to rebuild a better sample

    # Compute translational sample size
    summary_t_file_path = path.join(log_output_folder, "summary_t_errors")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_carmen_file_path,
                     relations_path=relations_re_file_path,
                     weights="{1, 1, 1, 0, 0, 0}",
                     log_path=path.join(log_output_folder, "summary_t.log"),
                     errors_path=summary_t_file_path)

    error_file = open(summary_t_file_path, "r")
    content = error_file.readlines()
    words = content[1].split(", ")
    std = float(words[1])
    var = math.pow(std, 2)
    z_a_2 = t.ppf(alpha, n_samples - 1)
    delta = max_error
    n_samples_t = int(math.pow(z_a_2, 2) * var / math.pow(delta, 2))

    # Compute rotational sample size
    summary_r_file_path = path.join(log_output_folder, "summary_r_errors")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_carmen_file_path,
                     relations_path=relations_re_file_path,
                     weights="{0, 0, 0, 1, 1, 1}",
                     log_path=path.join(log_output_folder, "summary_r.log"),
                     errors_path=summary_r_file_path)

    error_file = open(summary_r_file_path, "r")
    content = error_file.readlines()
    words = content[1].split(", ")
    std = float(words[1])
    var = math.pow(std, 2)
    z_a_2 = t.ppf(alpha, n_samples - 1)
    delta = max_error
    n_samples_r = int(math.pow(z_a_2, 2) * var / math.pow(delta, 2))

    # Select the biggest of the two
    n_samples = max(n_samples_t, n_samples_r)
    if n_samples < 10:
        print_error("n_samples too low", n_samples, n_samples_t, n_samples_r)
        return

    with open(relations_re_file_path, "w") as relations_file_re:
        for _ in range(n_samples):
            two_random_poses = interpolated_ground_truth_df[['t', 'x_gt', 'y_gt', 'theta_gt']].sample(n=2)
            t_1, x_1, y_1, theta_1 = two_random_poses.iloc[0]
            t_2, x_2, y_2, theta_2 = two_random_poses.iloc[1]

            # reorder data so that t_1 is before t_2
            if t_1 > t_2:
                # swap 1 and 2
                t_1, x_1, y_1, theta_1, t_2, x_2, y_2, theta_2 = t_2, x_2, y_2, theta_2, t_1, x_1, y_1, theta_1

            rel = get_matrix_diff((x_1, y_1, theta_1), (x_2, y_2, theta_2))

            x_relative = rel[0, 3]
            y_relative = rel[1, 3]
            theta_relative = math.atan2(rel[1, 0], rel[0, 0])

            relations_file_re.write("{t_1} {t_2} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(t_1=t_1, t_2=t_2, x=x_relative, y=y_relative, theta=theta_relative))

    relative_errors_dict['random_relations'] = dict()

    metric_evaluator_re_t_results_csv_path = path.join(log_output_folder, "re_t.csv")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_carmen_file_path,
                     relations_path=relations_re_file_path,
                     weights="{1, 1, 1, 0, 0, 0}",
                     log_path=path.join(log_output_folder, "re_t.log"),
                     errors_path=metric_evaluator_re_t_results_csv_path,
                     unsorted_errors_path=path.join(log_output_folder, "re_t_unsorted_errors"))

    metric_evaluator_re_t_results_df = pd.read_csv(metric_evaluator_re_t_results_csv_path, sep=', ', engine='python')
    relative_errors_dict['random_relations']['translation'] = dict()
    relative_errors_dict['random_relations']['translation']['mean'] = float(metric_evaluator_re_t_results_df['Mean'][0])
    relative_errors_dict['random_relations']['translation']['std'] = float(metric_evaluator_re_t_results_df['Std'][0])
    relative_errors_dict['random_relations']['translation']['min'] = float(metric_evaluator_re_t_results_df['Min'][0])
    relative_errors_dict['random_relations']['translation']['max'] = float(metric_evaluator_re_t_results_df['Max'][0])
    relative_errors_dict['random_relations']['translation']['n'] = float(metric_evaluator_re_t_results_df['NumMeasures'][0])

    metric_evaluator_re_r_results_csv_path = path.join(log_output_folder, "re_r.csv")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_carmen_file_path,
                     relations_path=relations_re_file_path,
                     weights="{0, 0, 0, 1, 1, 1}",
                     log_path=path.join(log_output_folder, "re_r.log"),
                     errors_path=metric_evaluator_re_r_results_csv_path,
                     unsorted_errors_path=path.join(log_output_folder, "re_r_unsorted_errors"))

    metric_evaluator_re_r_results_df = pd.read_csv(metric_evaluator_re_r_results_csv_path, sep=', ', engine='python')
    relative_errors_dict['random_relations']['rotation'] = dict()
    relative_errors_dict['random_relations']['rotation']['mean'] = float(metric_evaluator_re_r_results_df['Mean'][0])
    relative_errors_dict['random_relations']['rotation']['std'] = float(metric_evaluator_re_r_results_df['Std'][0])
    relative_errors_dict['random_relations']['rotation']['min'] = float(metric_evaluator_re_r_results_df['Min'][0])
    relative_errors_dict['random_relations']['rotation']['max'] = float(metric_evaluator_re_r_results_df['Max'][0])
    relative_errors_dict['random_relations']['rotation']['n'] = float(metric_evaluator_re_r_results_df['NumMeasures'][0])

    # sequential relations
    if compute_sequential_relations:
        ordered_relations_file_path = path.join(log_output_folder, "ordered_relations")
        with open(ordered_relations_file_path, "w") as relations_file_ordered:

            idx_delta = int(len(interpolated_ground_truth_df.index)/n_samples)
            if idx_delta == 0:
                idx_delta = 1

            for idx, first_row in interpolated_ground_truth_df.iloc[::idx_delta].iloc[0: -1].iterrows():
                second_row = interpolated_ground_truth_df.iloc[idx + idx_delta]

                rel = get_matrix_diff((first_row['x_gt'], first_row['y_gt'], first_row['theta_gt']), (second_row['x_gt'], second_row['y_gt'], second_row['theta_gt']))
                x_relative = rel[0, 3]
                y_relative = rel[1, 3]
                theta_relative = math.atan2(rel[1, 0], rel[0, 0])

                relations_file_ordered.write("{t_1} {t_2} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(t_1=first_row['t'], t_2=second_row['t'], x=x_relative, y=y_relative, theta=theta_relative))

        relative_errors_dict['sequential_relations'] = dict()

        metric_evaluator_ordered_t_results_csv_path = path.join(log_output_folder, "ordered_t.csv")
        metric_evaluator(exec_path=metric_evaluator_exec_path,
                         poses_path=estimated_poses_carmen_file_path,
                         relations_path=ordered_relations_file_path,
                         weights="{1, 1, 1, 0, 0, 0}",
                         log_path=path.join(log_output_folder, "ordered_t.log"),
                         errors_path=metric_evaluator_ordered_t_results_csv_path,
                         unsorted_errors_path=path.join(log_output_folder, "ordered_t_unsorted_errors"))

        metric_evaluator_ordered_t_results_df = pd.read_csv(metric_evaluator_ordered_t_results_csv_path, sep=', ', engine='python')
        relative_errors_dict['sequential_relations']['translation'] = dict()
        relative_errors_dict['sequential_relations']['translation']['mean'] = float(metric_evaluator_ordered_t_results_df['Mean'][0])
        relative_errors_dict['sequential_relations']['translation']['std'] = float(metric_evaluator_ordered_t_results_df['Std'][0])
        relative_errors_dict['sequential_relations']['translation']['min'] = float(metric_evaluator_ordered_t_results_df['Min'][0])
        relative_errors_dict['sequential_relations']['translation']['max'] = float(metric_evaluator_ordered_t_results_df['Max'][0])
        relative_errors_dict['sequential_relations']['translation']['n'] = float(metric_evaluator_ordered_t_results_df['NumMeasures'][0])

        metric_evaluator_ordered_r_results_csv_path = path.join(log_output_folder, "ordered_r.csv")
        metric_evaluator(exec_path=metric_evaluator_exec_path,
                         poses_path=estimated_poses_carmen_file_path,
                         relations_path=ordered_relations_file_path,
                         weights="{0, 0, 0, 1, 1, 1}",
                         log_path=path.join(log_output_folder, "ordered_r.log"),
                         errors_path=metric_evaluator_ordered_r_results_csv_path,
                         unsorted_errors_path=path.join(log_output_folder, "ordered_r_unsorted_errors"))

        metric_evaluator_ordered_r_results_df = pd.read_csv(metric_evaluator_ordered_r_results_csv_path, sep=', ', engine='python')
        relative_errors_dict['sequential_relations']['rotation'] = dict()
        relative_errors_dict['sequential_relations']['rotation']['mean'] = float(metric_evaluator_ordered_r_results_df['Mean'][0])
        relative_errors_dict['sequential_relations']['rotation']['std'] = float(metric_evaluator_ordered_r_results_df['Std'][0])
        relative_errors_dict['sequential_relations']['rotation']['min'] = float(metric_evaluator_ordered_r_results_df['Min'][0])
        relative_errors_dict['sequential_relations']['rotation']['max'] = float(metric_evaluator_ordered_r_results_df['Max'][0])
        relative_errors_dict['sequential_relations']['rotation']['n'] = float(metric_evaluator_ordered_r_results_df['NumMeasures'][0])

    return relative_errors_dict


def get_matrix_diff(p1, p2):
    """
    Computes the rototranslation difference of two points
    """

    x1, y1, theta1 = p1
    x2, y2, theta2 = p2

    m_translation1 = np.matrix(((1, 0, 0, x1),
                                (0, 1, 0, y1),
                                (0, 0, 1, 0),
                                (0, 0, 0, 1)))

    m_translation2 = np.matrix(((1, 0, 0, x2),
                                (0, 1, 0, y2),
                                (0, 0, 1, 0),
                                (0, 0, 0, 1)))

    m_rotation1 = np.matrix(((math.cos(theta1), -math.sin(theta1), 0, 0),
                             (math.sin(theta1), math.cos(theta1), 0, 0),
                             (0, 0, 1, 0),
                             (0, 0, 0, 1)))

    m_rotation2 = np.matrix(((math.cos(theta2), -math.sin(theta2), 0, 0),
                             (math.sin(theta2), math.cos(theta2), 0, 0),
                             (0, 0, 1, 0),
                             (0, 0, 0, 1)))

    m1 = m_translation1 * m_rotation1
    m2 = m_translation2 * m_rotation2
    return m1.I * m2


def absolute_localization_error_metrics(estimated_poses_file_path, ground_truth_poses_file_path):

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("absolute_localization_error_metrics: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(ground_truth_poses_file_path):
        print_error("absolute_localization_error_metrics: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return

    estimated_poses_df = pd.read_csv(estimated_poses_file_path)
    if len(estimated_poses_df.index) < 2:
        print_error("not enough estimated poses data points")
        return

    ground_truth_poses_df = pd.read_csv(ground_truth_poses_file_path)
    df = compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df)

    # if no matching ground truth data points are found, the metrics con not be computed
    if len(df.index) < 2:
        print_error("no matching ground truth data points were found")
        return

    absolute_error_of_each_segment = np.sqrt((df['x_est'] - df['x_gt'])**2 + (df['y_est'] - df['y_gt'])**2)

    absolute_error_dict = dict()
    absolute_error_dict['sum'] = float(absolute_error_of_each_segment.sum())
    absolute_error_dict['mean'] = float(absolute_error_of_each_segment.mean())
    return absolute_error_dict


# def env_passage_width(ground_truth_map, x, y):
#     from performance_modelling_py.environment.ground_truth_map import GroundTruthMap
#     ground_truth_map = GroundTruthMap()
#     vg = ground_truth_map.voronoi_graph
#     node_distances = vg.nodes["vertex"]
#     print(node_distances)
#     closest_node_index = np.argmin(node_distances)
#     return vg.nodes[closest_node_index]


def absolute_error_vs_voronoi_radius(estimated_poses_file_path, ground_truth_poses_file_path, ground_truth_map, samples_per_second=1.0):

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("absolute_localization_error_metrics: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(ground_truth_poses_file_path):
        print_error("absolute_localization_error_metrics: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return

    estimated_poses_df = pd.read_csv(estimated_poses_file_path)
    if len(estimated_poses_df.index) < 2:
        print_error("not enough estimated poses data points")
        return

    start_time = time.time()
    ground_truth_poses_df = pd.read_csv(ground_truth_poses_file_path)

    ground_truth_interpolated_poses_file_path = path.splitext(estimated_poses_file_path)[0] + "ground_truth_interpolated.csv"
    if path.exists(ground_truth_interpolated_poses_file_path):
        complete_trajectory_df = pd.read_csv(ground_truth_interpolated_poses_file_path)
    else:
        complete_trajectory_df = compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df)
        complete_trajectory_df.to_csv(ground_truth_interpolated_poses_file_path)

    run_duration = ground_truth_poses_df.iloc[-1]['t'] - ground_truth_poses_df.iloc[0]['t']
    resample_n = int(run_duration*samples_per_second)
    resample_ratio = max(1, len(complete_trajectory_df) // resample_n)
    trajectory_df = complete_trajectory_df.iloc[::resample_ratio].copy(deep=False)
    print_info("compute_ground_truth_interpolated_poses", time.time() - start_time)

    # if no matching ground truth data points are found, the metrics can not be computed
    if len(trajectory_df.index) < 2:
        print_error("no matching ground truth data points were found")
        return

    trajectory_points_list = list()
    trajectory_df['abs_err'] = np.sqrt((trajectory_df['x_est'] - trajectory_df['x_gt'])**2 + (trajectory_df['y_est'] - trajectory_df['y_gt'])**2)
    for _, row in trajectory_df.iterrows():
        x_gt, y_gt, abs_err = row['x_gt'], row['y_gt'], row['abs_err']
        trajectory_points_list.append(np.array([x_gt, y_gt]))

    start_time = time.time()
    min_radius = 4 * ground_truth_map.resolution
    voronoi_graph = ground_truth_map.voronoi_graph.subgraph(filter(
        lambda n: ground_truth_map.voronoi_graph.nodes[n]['radius'] >= min_radius,
        ground_truth_map.voronoi_graph.nodes
    ))
    print_info("ground_truth_map.voronoi_graph", time.time() - start_time)

    voronoi_vertices_list = list()
    voronoi_radii_list = list()
    for node_index, node_data in voronoi_graph.nodes.data():
        voronoi_vertices_list.append(node_data['vertex'])
        voronoi_radii_list.append(node_data['radius'])
    voronoi_vertices_array = np.array(voronoi_vertices_list)

    start_time = time.time()
    kdtree = scipy.spatial.cKDTree(voronoi_vertices_array)
    dist, indexes = kdtree.query(trajectory_points_list)
    print_info("kdtree", time.time() - start_time)

    trajectory_radii_list = list()
    for voronoi_vertex_index in indexes:
        trajectory_radii_list.append(voronoi_radii_list[voronoi_vertex_index])

    trajectory_df['trajectory_radius'] = trajectory_radii_list

    # # plotting
    # import matplotlib.pyplot as plt
    #
    # # fig, ax = plt.subplots()
    # # ax.scatter(trajectory_df['trajectory_radius'], trajectory_df['abs_err'])  # , color='red', s=4.0, marker='o')
    # # plt.xlabel("trajectory_radius")
    # # plt.ylabel("abs_err")
    # # plt.show()
    # # plt.cla()
    # #
    # # plt.plot(trajectory_df['t'], trajectory_df['abs_err'])
    # # plt.plot(trajectory_df['t'], trajectory_df['trajectory_radius'])
    # # plt.legend()
    # # plt.show()
    # # plt.cla()
    #
    # bins = np.linspace(0, 3, 4*5)
    # plt.hist(trajectory_df['abs_err'], bins=bins)
    # plt.hist(trajectory_df[(trajectory_df['trajectory_radius'] > 0.0) & (trajectory_df['trajectory_radius'] < 3.5)]['abs_err'], bins=bins)
    # plt.title("0.0 ~ 3.5")
    # plt.show()
    # plt.cla()
    #
    # plt.hist(trajectory_df['abs_err'], bins=bins)
    # plt.hist(trajectory_df[(trajectory_df['trajectory_radius'] > 3.5) & (trajectory_df['trajectory_radius'] < 5.0)]['abs_err'], bins=bins)
    # plt.title("3.5 ~ 5.0")
    # plt.show()
    # plt.cla()
    #
    # plt.hist(trajectory_df['abs_err'], bins=bins)
    # plt.hist(trajectory_df[(trajectory_df['trajectory_radius'] > 0.0) & (trajectory_df['trajectory_radius'] < 1.0)]['abs_err'], bins=bins)
    # plt.title("0.0 ~ 1.0")
    # plt.show()
    # plt.cla()
    #
    # bins = np.linspace(0, 9, 8*4)
    # plt.hist(trajectory_df['trajectory_radius'], bins=bins)
    # plt.hist(trajectory_df[(trajectory_df['abs_err'] > .3) & (trajectory_df['abs_err'] < .5)]['trajectory_radius'], bins=bins)
    # plt.show()
    #
    # # plot trajectory and distances from voronoi vertices
    # distance_segments = list()
    # for trajectory_point_index, voronoi_vertex_index in enumerate(indexes):
    #     voronoi_vertex = voronoi_vertices_array[voronoi_vertex_index]
    #     trajectory_point = trajectory_points_list[trajectory_point_index]
    #     distance_segments.append((voronoi_vertex, trajectory_point))
    #
    # trajectory_segments = zip(trajectory_points_list[0: -1], trajectory_points_list[1:])
    # ground_truth_map.save_voronoi_plot_and_trajectory("~/tmp/vg_and_traj.svg", trajectory_segments + distance_segments)

    return trajectory_df


def absolute_error_vs_scan_range(estimated_poses_file_path, ground_truth_poses_file_path, scans_file_path, samples_per_second=1.0):

    # check required files exist
    if not path.isfile(estimated_poses_file_path):
        print_error("absolute_localization_error_metrics: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(ground_truth_poses_file_path):
        print_error("absolute_localization_error_metrics: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return

    if not path.isfile(scans_file_path):
        print_error("absolute_localization_error_metrics: scans_file_path file not found {}".format(scans_file_path))
        return

    start_time = time.time()
    estimated_poses_df = pd.read_csv(estimated_poses_file_path)
    if len(estimated_poses_df.index) < 2:
        print_error("not enough estimated poses data points")
        return
    print_info("estimated_poses_df", time.time() - start_time)

    with open(scans_file_path) as scans_file:
        scan_lines = scans_file.read().split('\n')

    start_time = time.time()
    scans_df = pd.DataFrame(columns=['t', 'min_range', 'max_range', 'median_range', 'num_valid_ranges'])
    for scan_line in scan_lines:
        scan_fields = scan_line.split(', ')
        if len(scan_fields) > 1:
            t, angle_min, angle_max, angle_increment, range_min, range_max = map(float, scan_fields[0:6])
            ranges = map(float, scan_fields[6:])
            num_valid_ranges = sum(map(lambda r: range_min < r < range_max, ranges))
            record = {
                't': t,
                'min_range': min(ranges),
                'max_range': max(ranges),
                'median_range': np.median(ranges),
                'num_valid_ranges': num_valid_ranges,
                'sensor_range_min': range_min,
                'sensor_range_max': range_max,
                'num_ranges': len(ranges),
                'angle_increment': angle_increment,
                'fov_rad': angle_max - angle_min
            }
            scans_df = scans_df.append(record, ignore_index=True)
    print_info("scans_df", time.time() - start_time)

    # print(scans_df)

    start_time = time.time()
    ground_truth_poses_df = pd.read_csv(ground_truth_poses_file_path)
    print_info("ground_truth_poses_df", time.time() - start_time)

    start_time = time.time()
    ground_truth_interpolated_poses_file_path = path.splitext(estimated_poses_file_path)[0] + "ground_truth_interpolated.csv"
    if path.exists(ground_truth_interpolated_poses_file_path):
        complete_trajectory_df = pd.read_csv(ground_truth_interpolated_poses_file_path)
    else:
        complete_trajectory_df = compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df)
        complete_trajectory_df.to_csv(ground_truth_interpolated_poses_file_path)

    run_duration = ground_truth_poses_df.iloc[-1]['t'] - ground_truth_poses_df.iloc[0]['t']
    resample_n = int(run_duration*samples_per_second)
    resample_ratio = max(1, len(complete_trajectory_df) // resample_n)
    trajectory_df = complete_trajectory_df.iloc[::resample_ratio].copy(deep=False)
    trajectory_df['abs_err'] = np.sqrt((trajectory_df['x_est'] - trajectory_df['x_gt'])**2 + (trajectory_df['y_est'] - trajectory_df['y_gt'])**2)
    # print(trajectory_df)
    print_info("trajectory_df", time.time() - start_time)

    start_time = time.time()
    merge_tolerance = 0.25
    tolerance = pd.Timedelta('{}s'.format(merge_tolerance))
    trajectory_df['t_datetime'] = pd.to_datetime(trajectory_df['t'], unit='s')
    scans_df['t_datetime'] = pd.to_datetime(scans_df['t'], unit='s')
    near_matches_df = pd.merge_asof(
        left=scans_df,
        right=trajectory_df,
        on='t_datetime',
        direction='nearest',
        tolerance=tolerance,
        suffixes=('_scan', '_gt')
    )
    trajectory_and_scan_df = near_matches_df[(pd.notnull(near_matches_df['t_scan'])) & (pd.notnull(near_matches_df['t_gt']))].copy(deep=False)
    print_info("trajectory_and_scan_df", time.time() - start_time)

    # import matplotlib.pyplot as plt
    # plt.scatter(trajectory_and_scan_df['min_range'], trajectory_and_scan_df['abs_err'])  # , color='red', s=4.0, marker='o')
    # plt.xlabel("min_range")
    # plt.ylabel("abs_err")
    # plt.show()
    # plt.cla()
    #
    # plt.plot(trajectory_and_scan_df['t_gt'], trajectory_and_scan_df['abs_err'])
    # plt.plot(trajectory_and_scan_df['t_scan'], trajectory_and_scan_df['min_range']/trajectory_and_scan_df['sensor_range_max'])
    # plt.plot(trajectory_and_scan_df['t_scan'], trajectory_and_scan_df['num_valid_ranges']/trajectory_and_scan_df['num_ranges'])
    # plt.legend()
    # plt.show()
    # plt.cla()

    return trajectory_and_scan_df


def trajectory_length_metric(ground_truth_poses_file_path):

    # check required files exist
    if not path.isfile(ground_truth_poses_file_path):
        print_error("compute_trajectory_length: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return None

    df = pd.read_csv(ground_truth_poses_file_path)

    squared_deltas = (df[['x', 'y']] - df[['x', 'y']].shift(periods=1)) ** 2  # equivalent to (x_2-x_1)**2, (y_2-y_1)**2, for each row
    sum_of_squared_deltas = squared_deltas['x'] + squared_deltas['y']  # equivalent to (x_2-x_1)**2 + (y_2-y_1)**2, for each row
    euclidean_distance_of_deltas = np.sqrt(sum_of_squared_deltas)  # equivalent to sqrt( (x_2-x_1)**2 + (y_2-y_1)**2 ), for each row
    trajectory_length = euclidean_distance_of_deltas.sum()

    return float(trajectory_length)


def absolute_error_vs_geometric_similarity(estimated_poses_file_path, ground_truth_poses_file_path, ground_truth_map, horizon_length=3.5, samples_per_second=10.0, max_iterations=20):
    # import matplotlib.pyplot as plt
    from icp import iterative_closest_point
    from skimage.draw import line, circle_perimeter
    # plt.rcParams["figure.figsize"] = (10, 10)

    start_time = time.time()
    estimated_poses_df = pd.read_csv(estimated_poses_file_path)
    if len(estimated_poses_df.index) < 2:
        print_error("not enough estimated poses data points")
        return
    print_info("estimated_poses_df", time.time() - start_time)

    start_time = time.time()
    ground_truth_poses_df = pd.read_csv(ground_truth_poses_file_path)
    print_info("ground_truth_poses_df", time.time() - start_time)

    start_time = time.time()
    ground_truth_interpolated_poses_file_path = path.splitext(estimated_poses_file_path)[0] + "ground_truth_interpolated.csv"
    if path.exists(ground_truth_interpolated_poses_file_path):
        print_info("using cached ground_truth_interpolated_poses [{}]".format(ground_truth_interpolated_poses_file_path))
        complete_trajectory_df = pd.read_csv(ground_truth_interpolated_poses_file_path)
    else:
        complete_trajectory_df = compute_ground_truth_interpolated_poses(estimated_poses_df, ground_truth_poses_df)
        complete_trajectory_df.to_csv(ground_truth_interpolated_poses_file_path)

    run_duration = ground_truth_poses_df.iloc[-1]['t'] - ground_truth_poses_df.iloc[0]['t']
    resample_n = int(run_duration*samples_per_second)
    resample_ratio = max(1, len(complete_trajectory_df) // resample_n)
    trajectory_df = complete_trajectory_df.iloc[::resample_ratio].copy(deep=False)
    trajectory_df['abs_err'] = np.sqrt((trajectory_df['x_est'] - trajectory_df['x_gt'])**2 + (trajectory_df['y_est'] - trajectory_df['y_gt'])**2)
    print_info("trajectory_df", time.time() - start_time)

    delta_length = 0.2  # displacement of x and x_prime
    ic = ground_truth_map.map_frame_to_image_coordinates
    mf = ground_truth_map.image_to_map_frame_coordinates

    map_image_array = np.array(ground_truth_map.map_image.convert(mode="L")).transpose()
    occupancy_grid = map_image_array == 0

    translation_score_0_column = list()
    translation_score_column = list()
    rotation_score_0_column = list()
    rotation_score_column = list()
    plot_lines = list()
    icp_time = 0
    ray_tracing_time = 0

    rows_count = len(trajectory_df[['x_gt', 'y_gt']])
    progress_count = 0

    for i, row_df in trajectory_df[['x_gt', 'y_gt']].iterrows():
        x_mf = np.array(row_df)
        p_x, p_y = ic(x_mf)

        ray_tracing_start_time = time.time()
        visible_points_x = set()
        perimeter_points = np.array(circle_perimeter(p_x, p_y, int((horizon_length + delta_length)/ground_truth_map.resolution), method='andres')).transpose()
        for perimeter_x, perimeter_y in perimeter_points:

            ray_points = np.array(line(p_x, p_y, perimeter_x, perimeter_y)).transpose()
            for ray_x, ray_y in ray_points:
                if occupancy_grid[ray_x, ray_y]:
                    visible_points_x.add((ray_x, ray_y))
                    break

        visible_points_x_mf = np.array(list(map(mf, visible_points_x)))
        if len(visible_points_x_mf) == 0:
            translation_score_0_column.append(np.nan)
            rotation_score_0_column.append(np.nan)
            translation_score_column.append(np.nan)
            rotation_score_column.append(np.nan)
            continue

        visible_points_x_mf_o = visible_points_x_mf - x_mf
        ray_tracing_time += time.time() - ray_tracing_start_time

        delta_trans_list = delta_length * np.array([
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (.7, .7),
            (.7, -.7),
            (-.7, .7),
            (-.7, -.7),
        ])

        delta_theta_list = [-0.1, -0.05, 0.05, 0.1]

        translation_score_0_list = list()
        translation_score_list = list()
        for delta_trans in delta_trans_list:
            x_prime_mf = x_mf + delta_trans
            p_x_prime, p_y_prime = ic(x_prime_mf)

            ray_tracing_start_time = time.time()
            visible_points_x_prime = set()
            perimeter_points_prime = np.array(circle_perimeter(p_x_prime, p_y_prime, int(horizon_length / ground_truth_map.resolution), method='andres')).transpose()
            for perimeter_x_prime, perimeter_y_prime in perimeter_points_prime:

                ray_points_prime = np.array(line(p_x_prime, p_y_prime, perimeter_x_prime, perimeter_y_prime)).transpose()
                for ray_x_prime, ray_y_prime in ray_points_prime:
                    if occupancy_grid[ray_x_prime, ray_y_prime]:
                        visible_points_x_prime.add((ray_x_prime, ray_y_prime))
                        break

            visible_points_x_prime_mf = np.array(list(map(mf, visible_points_x_prime)))
            ray_tracing_time += time.time() - ray_tracing_start_time

            if len(visible_points_x_prime_mf) == 0:
                continue

            visible_points_x_prime_mf_o = visible_points_x_prime_mf - x_prime_mf

            start_time = time.time()
            transform, scores, transformed_points_prime, num_iterations = iterative_closest_point(visible_points_x_prime_mf_o, visible_points_x_mf_o, max_iterations=max_iterations, tolerance=0.001)

            translation_score_0_list.append(scores[0])

            translation = transform[:2, 2]
            translation_score = float(np.sqrt(np.sum((translation - delta_trans) ** 2)) / np.sqrt(np.sum(delta_trans ** 2)))

            # print("translation score: {s:1.4f} \t translation score_0: {s0:1.4f} \t i: {i}".format(s=translation_score, s0=scores[0], i=num_iterations))

            translation_score_list.append(translation_score)
            icp_time += time.time() - start_time

            # if translation_score > 0.5:
            #     plt.scatter(*visible_points_x_mf_o.transpose(), color='black', marker='x')
            #     plt.scatter(*visible_points_x_prime_mf_o.transpose(), color='red', marker='x')
            #     plt.scatter(*transformed_points_prime.transpose(), color='blue', marker='x')
            #     hl = horizon_length + delta_length
            #     plt.xlim(-hl, hl)
            #     plt.ylim(-hl, hl)
            #     plt.gca().set_aspect('equal', adjustable='box')
            #     plt.show()

            plot_lines.append((x_mf, x_prime_mf, translation_score))

        if len(translation_score_0_list) == 0:
            translation_score_0_column.append(np.nan)
            translation_score_column.append(np.nan)
        else:
            translation_score_0_column.append(np.mean(translation_score_0_list))
            translation_score_column.append(np.mean(translation_score_list))

        rotation_score_0_list = list()
        rotation_score_list = list()
        for delta_theta in delta_theta_list:
            rot_mat = np.array([
                [np.cos(delta_theta), np.sin(delta_theta)],
                [-np.sin(delta_theta), np.cos(delta_theta)]
            ])
            visible_points_x_prime_mf_o = np.dot(visible_points_x_mf_o, rot_mat)

            start_time = time.time()
            transform, scores, transformed_points_prime, num_iterations = iterative_closest_point(visible_points_x_prime_mf_o, visible_points_x_mf_o, max_iterations=max_iterations, tolerance=0.001)

            rotation_score_0_list.append(scores[0])

            rotation_mat = transform[:2, :2].T
            rotation = math.atan2(rotation_mat[0, 1], rotation_mat[0, 0])
            rotation_score = float(np.fabs(rotation + delta_theta)) / np.fabs(delta_theta)

            # print("rotation_score: {s:1.4f} \t rotation_score_0: {s0:1.4f} \t i: {i} \t transform angle: {r:1.4f} \t delta theta: {dt:1.4f}".format(s=rotation_score, s0=scores[0], i=num_iterations, r=rotation, dt=delta_theta))

            rotation_score_list.append(rotation_score)
            icp_time += time.time() - start_time

            # if rotation_score > 0.5:
            #     plt.scatter(*visible_points_x_mf_o.transpose(), color='black', marker='x')
            #     plt.scatter(*visible_points_x_prime_mf_o.transpose(), color='red', marker='x')
            #     plt.scatter(*transformed_points_prime.transpose(), color='blue', marker='x')
            #     hl = horizon_length + delta_length
            #     plt.xlim(-hl, hl)
            #     plt.ylim(-hl, hl)
            #     plt.gca().set_aspect('equal', adjustable='box')
            #     plt.show()

            # plot_lines.append((x_mf, x_prime_mf, translation_score))

        if len(rotation_score_0_list) == 0:
            rotation_score_0_column.append(np.nan)
            rotation_score_column.append(np.nan)
        else:
            rotation_score_0_column.append(np.mean(rotation_score_0_list))
            rotation_score_column.append(np.mean(rotation_score_list))

        progress_count += 1
        print_info(int(progress_count / float(rows_count) * 100), "%", replace_previous_line=True)

    trajectory_df['translation_score_0'] = translation_score_0_column
    trajectory_df['translation_score'] = translation_score_column
    trajectory_df['rotation_score_0'] = rotation_score_0_column
    trajectory_df['rotation_score'] = rotation_score_column

    # print(trajectory_df)

    print_info("icp_time", icp_time)
    print_info("ray_tracing_time", ray_tracing_time)

    # for (x_0, y_0), (x_1, y_1), score in plot_lines:
    #     plt.plot([x_0, x_1], [y_0, y_1], linewidth=3*score, color='black')
    # plt.plot([trajectory_df['x_gt'], trajectory_df['x_est']], [trajectory_df['y_gt'], trajectory_df['y_est']], color='red')
    # plt.show()

    # plt.plot(trajectory_df['t'], trajectory_df['abs_err'], label='abs_err')
    # plt.plot(trajectory_df['t'], trajectory_df['rotation_score_0'], label='rotation_score_0')
    # plt.legend()
    # plt.show()
    #
    # plt.scatter(trajectory_df['rotation_score_0'], trajectory_df['abs_err'])
    # plt.xlabel("rotation_score_0")
    # plt.ylabel("abs_err")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(trajectory_df['t'], trajectory_df['abs_err'], label='abs_err')
    # plt.plot(trajectory_df['t'], trajectory_df['rotation_score'], label='rotation_score')
    # plt.legend()
    # plt.show()
    #
    # plt.scatter(trajectory_df['rotation_score'], trajectory_df['abs_err'], label='abs_err vs rotation_score')
    # plt.xlabel("rotation_score")
    # plt.ylabel("abs_err")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(trajectory_df['t'], trajectory_df['abs_err'], label='abs_err')
    # plt.plot(trajectory_df['t'], trajectory_df['translation_score_0'], label='translation_score_0')
    # plt.plot(trajectory_df['t'], trajectory_df['translation_score'], label='translation_score')
    # plt.legend()
    # plt.show()
    #
    # plt.scatter(trajectory_df['translation_score_0'], trajectory_df['abs_err'], label='abs_err vs translation_score_0')
    # plt.xlabel("translation_score_0")
    # plt.ylabel("abs_err")
    # plt.legend()
    # plt.show()
    #
    # plt.scatter(trajectory_df['translation_score'], trajectory_df['abs_err'], label='abs_err vs translation_score')
    # plt.xlabel("translation_score")
    # plt.ylabel("abs_err")
    # plt.legend()
    # plt.show()

    return trajectory_df


# def test_metrics(run_output_folder):
#
#     run_info_path = path.join(run_output_folder, "run_info.yaml")
#     if not path.exists(run_info_path) or not path.isfile(run_info_path):
#         print_error("run info file does not exists")
#
#     with open(run_info_path) as run_info_file:
#         run_info = yaml.safe_load(run_info_file)
#
#     environment_folder = run_info['environment_folder']
#     metrics_result_folder_path = path.join(run_output_folder, "metric_results")
#     ground_truth_map_info_path = path.join(environment_folder, "data", "map.yaml")
#     ground_truth_map = GroundTruthMap(ground_truth_map_info_path)
#
#     # localization metrics
#     estimated_poses_path = path.join(run_output_folder, "benchmark_data", "estimated_poses.csv")
#     ground_truth_poses_path = path.join(run_output_folder, "benchmark_data", "ground_truth_poses.csv")
#
#     absolute_error_vs_geometric_similarity(estimated_poses_path, ground_truth_poses_path, ground_truth_map).to_csv(path.join(metrics_result_folder_path, "absolute_error_vs_geometric_similarity.csv"))
#
#
# if __name__ == '__main__':
#     # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/elysium/performance_modelling/output/localization/*"))))
#     # for progress, run_folder in enumerate(run_folders):
#     #     print_info("main: compute_metrics {}% {}".format((progress + 1)*100/len(run_folders), run_folder))
#     #     test_metrics(path.expanduser(run_folder))
#
#     run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_localization/run_000000000/"))))
#     # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_slam/run_000000000/"))))
#     # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/elysium/performance_modelling/output/localization/*"))))
#     # run_folders = list(filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling/output/test_localization/*"))))
#     last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
#     print("last run folder:", last_run_folder)
#     test_metrics(last_run_folder)
