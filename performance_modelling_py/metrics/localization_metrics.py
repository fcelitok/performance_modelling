#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import math
import numpy as np
import pandas as pd
import os
import yaml
from os import path
from scipy.stats import t
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

    interpolated_ground_truth_df = pd.DataFrame(columns=['t', 'x', 'y', 'theta'])
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
