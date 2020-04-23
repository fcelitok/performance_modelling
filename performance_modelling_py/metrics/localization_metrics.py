#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import math
import numpy as np
import pandas as pd
import os
import random
import yaml
from collections import deque
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


def relative_localization_error_metrics(log_output_folder, estimated_poses_file_path, ground_truth_poses_file_path, alpha=0.99, max_error=0.02):
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

    # compute acceptable ground truth relation times
    ground_truth_dict = dict()
    with open(ground_truth_poses_file_path, "r") as ground_truth_file:
        for line in ground_truth_file:
            time, x, y, theta = map(float, line.split(', '))
            ground_truth_dict[time] = (x, y, theta)
    ground_truth_times = sorted(ground_truth_dict.keys())
    if len(ground_truth_times) < 2:
        return

    estimated_poses_dict = dict()
    with open(estimated_poses_file_path, "r") as estimated_poses_file:
        for line in estimated_poses_file:
            x, y, theta, time = map(float, line.split(' ')[-4:])
            estimated_poses_dict[time] = (x, y, theta)
    estimated_poses_times = deque(sorted(estimated_poses_dict.keys()))
    if len(estimated_poses_times) < 2:
        return

    # only accept the ground truth values that are close in time to the estimated_poses values
    max_diff = 0.1
    ground_truth_acceptable_times = list()
    for i in ground_truth_times:
        while len(estimated_poses_times) > 0 and estimated_poses_times[0] - i <= -max_diff:  # discard all estimated_poses times that are too old
            estimated_poses_times.popleft()

        if len(estimated_poses_times) == 0:  # if all estimated_poses values have been discarded because too old, no more ground truth values can be accepted
            break

        # once old estimated_poses values are discarded, either the next value is acceptable or too young
        if estimated_poses_times[0] - i < max_diff:  # accept the gt value if the difference is within range: -max_diff <= estimated_poses_times[0] - i <= max_diff
            ground_truth_acceptable_times.append(i)

    # random relations
    relations_re_file_path = path.join(log_output_folder, "re_relations")
    with open(relations_re_file_path, "w") as relations_file_re:

        if len(ground_truth_acceptable_times) == 0:
            return

        n_samples = 500
        for _ in range(n_samples):
            first_stamp = float(random.choice(ground_truth_acceptable_times))
            second_stamp = float(random.choice(ground_truth_acceptable_times))
            if first_stamp > second_stamp:
                first_stamp, second_stamp = second_stamp, first_stamp
            first_pos = ground_truth_dict[first_stamp]
            second_pos = ground_truth_dict[second_stamp]

            rel = get_matrix_diff(first_pos, second_pos)

            x = rel[0, 3]
            y = rel[1, 3]
            theta = math.atan2(rel[1, 0], rel[0, 0])

            relations_file_re.write("{first_stamp} {second_stamp} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(first_stamp=first_stamp, second_stamp=second_stamp, x=x, y=y, theta=theta))

    # Run the metric evaluator on this relations file, read the sample standard deviation and exploit it to rebuild a better sample

    # Compute translational sample size
    summary_t_file_path = path.join(log_output_folder, "summary_t_errors")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_file_path,
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
                     poses_path=estimated_poses_file_path,
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
        return

    with open(relations_re_file_path, "w") as relations_file_re:
        for _ in range(n_samples):
            first_stamp = float(random.choice(ground_truth_acceptable_times))
            second_stamp = float(random.choice(ground_truth_acceptable_times))
            if first_stamp > second_stamp:
                first_stamp, second_stamp = second_stamp, first_stamp
            first_pos = ground_truth_dict[first_stamp]
            second_pos = ground_truth_dict[second_stamp]

            rel = get_matrix_diff(first_pos, second_pos)
            x = rel[0, 3]
            y = rel[1, 3]
            theta = math.atan2(rel[1, 0], rel[0, 0])

            relations_file_re.write("{first_stamp} {second_stamp} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(first_stamp=first_stamp, second_stamp=second_stamp, x=x, y=y, theta=theta))

    relative_errors_dict['random_relations'] = dict()

    metric_evaluator_re_t_results_csv_path = path.join(log_output_folder, "re_t.csv")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_file_path,
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
                     poses_path=estimated_poses_file_path,
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

    # ordered relations
    ordered_relations_file_path = path.join(log_output_folder, "ordered_relations")
    with open(ordered_relations_file_path, "w") as relations_file_ordered:

        idx_delta = len(ground_truth_acceptable_times)/n_samples
        if idx_delta == 0:
            idx_delta = 1

        for idx, first_stamp in enumerate(ground_truth_acceptable_times[0::idx_delta][0:-1]):
            second_stamp = ground_truth_acceptable_times[idx + idx_delta]

            first_pos = ground_truth_dict[first_stamp]
            second_pos = ground_truth_dict[second_stamp]

            rel = get_matrix_diff(first_pos, second_pos)
            x = rel[0, 3]
            y = rel[1, 3]
            theta = math.atan2(rel[1, 0], rel[0, 0])

            relations_file_ordered.write("{first_stamp} {second_stamp} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(first_stamp=first_stamp, second_stamp=second_stamp, x=x, y=y, theta=theta))

    relative_errors_dict['sequential_relations'] = dict()

    metric_evaluator_ordered_t_results_csv_path = path.join(log_output_folder, "ordered_t.csv")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=estimated_poses_file_path,
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
                     poses_path=estimated_poses_file_path,
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
        print_error("compute_relative_localization_error: estimated_poses file not found {}".format(estimated_poses_file_path))
        return

    if not path.isfile(ground_truth_poses_file_path):
        print_error("compute_relative_localization_error: ground_truth_poses file not found {}".format(ground_truth_poses_file_path))
        return

    absolute_error_dict = dict()

    # compute matching ground truth and estimated poses times
    ground_truth_dict = dict()
    with open(ground_truth_poses_file_path, "r") as ground_truth_file:
        for line in ground_truth_file:
            time, x, y, theta = map(float, line.split(', '))
            ground_truth_dict[time] = (x, y, theta)

    estimated_poses_dict = dict()
    with open(estimated_poses_file_path, "r") as estimated_poses_file:
        for line in estimated_poses_file:
            x, y, theta, time = map(float, line.split(' ')[-4:])
            estimated_poses_dict[time] = (x, y, theta)

    matching_poses_dict = dict()
    for time in ground_truth_dict.keys():
        if time in estimated_poses_dict:
            matching_poses_dict[time] = (estimated_poses_dict[time], ground_truth_dict[time])

    def euclidean_distance(poses):
        a, b = poses
        a_x, a_y, _ = a
        b_x, b_y, _ = b

        return np.sqrt(np.sum((np.array((a_x, a_y)) - np.array((b_x, b_y)))**2))

    absolute_errors_list = map(euclidean_distance, matching_poses_dict.values())
    absolute_error_dict['sum'] = float(sum(absolute_errors_list))
    absolute_error_dict['mean'] = float(sum(absolute_errors_list)/len(absolute_errors_list))

    return absolute_error_dict


def trajectory_length_metric(ground_truth_file_path):

    # check required files exist
    if not path.isfile(ground_truth_file_path):
        print_error("compute_trajectory_length: ground_truth_poses file not found {}".format(ground_truth_file_path))
        return None

    # compute matching ground truth and estimated poses times
    ground_truth_points = list()
    with open(ground_truth_file_path, "r") as ground_truth_file:
        for line in ground_truth_file:
            _, x, y, _ = map(float, line.split(', '))
            ground_truth_points.append((x, y))

    def euclidean_distance(a, b):
        a_x, a_y = a
        b_x, b_y = b

        return np.sqrt(np.sum((np.array((a_x, a_y)) - np.array((b_x, b_y)))**2))

    trajectory_length = 0.0

    for i in range(len(ground_truth_points)-1):
        trajectory_length += euclidean_distance(ground_truth_points[i], ground_truth_points[i+1])

    return float(trajectory_length)


def compute_localization_metrics(run_output_folder):
    """
    Given a run folder path, compute relation files and localisation metric results
    """
    estimated_correction_poses_path = path.join(run_output_folder, "benchmark_data", "base_link_correction_poses")
    estimated_poses_path = path.join(run_output_folder, "benchmark_data", "base_link_poses")
    ground_truth_poses_path = path.join(run_output_folder, "benchmark_data", "ground_truth_poses")

    logs_folder_path = path.join(run_output_folder, "logs")
    metrics_result_folder_path = path.join(run_output_folder, "metric_results")
    metrics_result_file_path = path.join(metrics_result_folder_path, "localisation_metrics.yaml")

    metrics_result_dict = dict()

    metrics_result_dict['trajectory_length'] = trajectory_length_metric(ground_truth_poses_path)

    metrics_result_dict['relative_localization_correction_error'] = relative_localization_error_metrics(path.join(logs_folder_path, "relative_localisation_correction_error"), estimated_correction_poses_path, ground_truth_poses_path)
    metrics_result_dict['relative_localization_error'] = relative_localization_error_metrics(path.join(logs_folder_path, "relative_localisation_error"), estimated_poses_path, ground_truth_poses_path)

    metrics_result_dict['absolute_localization_correction_error'] = absolute_localization_error_metrics(estimated_correction_poses_path, ground_truth_poses_path)
    metrics_result_dict['absolute_localization_error'] = absolute_localization_error_metrics(estimated_poses_path, ground_truth_poses_path)

    if not path.exists(metrics_result_folder_path):
        os.makedirs(metrics_result_folder_path)

    with open(metrics_result_file_path, 'w') as metrics_result_file:
        yaml.dump(metrics_result_dict, metrics_result_file, default_flow_style=False)


if __name__ == '__main__':
    run_folders = filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling_output/test_1/*")))
    # last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
    # print("last run folder:", last_run_folder)
    for progress, run_folder in enumerate(run_folders):
        print_info("main: compute_localization_metrics {}% {}".format((progress + 1)*100/len(run_folders), run_folder))
        compute_localization_metrics(path.expanduser(run_folder))
