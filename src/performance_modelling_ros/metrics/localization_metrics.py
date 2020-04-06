#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import math
import numpy as np
import os
import random
from collections import deque
from os import path
from scipy.stats import t
import subprocess

import roslib
import roslib.packages

from performance_modelling_ros.utils import print_info, print_error


def metric_evaluator(exec_path, poses_path, relations_path, weights, log_path, errors_path, unsorted_errors_path=None):
    with open(log_path, 'w') as stdout_log_file:
        if unsorted_errors_path is None:
            p = subprocess.Popen([exec_path, "-s", poses_path, "-r", relations_path, "-w", weights, "-e", errors_path], stdout=stdout_log_file)
        else:
            p = subprocess.Popen([exec_path, "-s", poses_path, "-r", relations_path, "-w", weights, "-e", errors_path, "-eu", unsorted_errors_path], stdout=stdout_log_file)
        p.wait()


def compute_relative_localization_error(results_output_folder, log_output_folder, base_link_poses_file_path, ground_truth_file_path, alpha=0.99, max_error=0.02):
    """
    Generates the ordered and the random relations files and computes the metrics
    """

    # prepare folder structure
    if not path.exists(results_output_folder):
        os.makedirs(results_output_folder)

    if not path.exists(log_output_folder):
        os.makedirs(log_output_folder)

    # check required files exist
    if not path.isfile(base_link_poses_file_path):
        print_error("compute_relative_localization_error: base_link_poses_file_path file not found {}".format(base_link_poses_file_path))
        return

    if not path.isfile(ground_truth_file_path):
        print_error("compute_relative_localization_error: ground_truth_poses file not found {}".format(ground_truth_file_path))
        return

    # find the metricEvaluator executable
    metric_evaluator_package_name = 'performance_modelling'
    metric_evaluator_exec_name = 'metricEvaluator'
    metric_evaluator_resources_list = roslib.packages.find_resource(metric_evaluator_package_name, metric_evaluator_exec_name)
    if len(metric_evaluator_resources_list) > 1:
        print_error("compute_relative_localization_error: multiple files named [{resource_name}}] in package [{package_name}]:%s".format(resource_name=metric_evaluator_exec_name, package_name=metric_evaluator_package_name))
        return
    elif len(metric_evaluator_resources_list) == 0:
        print_error("compute_relative_localization_error: no files named [{resource_name}}] in package [{package_name}]:%s".format(resource_name=metric_evaluator_exec_name, package_name=metric_evaluator_package_name))
        return
    metric_evaluator_exec_path = metric_evaluator_resources_list[0]

    # compute acceptable ground truth relation times
    print_info("compute_relative_localization_error: computing acceptable ground truth relation times")
    ground_truth_dict = dict()
    with open(ground_truth_file_path, "r") as ground_truth_file:
        for line in ground_truth_file:
            time, x, y, theta = map(float, line.split(', '))
            ground_truth_dict[time] = (x, y, theta)
    ground_truth_times = sorted(ground_truth_dict.keys())
    if len(ground_truth_times) < 2:
        return

    base_link_dict = dict()
    with open(base_link_poses_file_path, "r") as base_link_file:
        for line in base_link_file:
            x, y, theta, time = map(float, line.split(' ')[-4:])
            base_link_dict[time] = (x, y, theta)
    base_link_times = deque(sorted(base_link_dict.keys()))
    if len(base_link_times) < 2:
        return

    # only accept the ground truth values that are close in time to the base_link values
    max_diff = 0.1
    ground_truth_acceptable_times = list()
    for i in ground_truth_times:
        while len(base_link_times) > 0 and base_link_times[0] - i <= -max_diff:  # discard all base_link times that are too old
            base_link_times.popleft()

        if len(base_link_times) == 0:  # if all base_link values have been discarded because too old, no more ground truth values can be accepted
            break

        # once old base_link values are discarded, either the next value is acceptable or too young
        if base_link_times[0] - i < max_diff:  # accept the gt value if the difference is within range: -max_diff <= base_link_times[0] - i <= max_diff
            ground_truth_acceptable_times.append(i)

    # random relations
    print_info("compute_relative_localization_error: computing random relations")
    relations_re_file_path = path.join(results_output_folder, "re_relations")
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
    print_info("compute_relative_localization_error: computing metric summary_t")
    summary_t_file_path = path.join(results_output_folder, "summary_t_errors")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=base_link_poses_file_path,
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
    print_info("compute_relative_localization_error: computing metric summary_r")
    summary_r_file_path = path.join(results_output_folder, "summary_r_errors")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=base_link_poses_file_path,
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
        print_error("compute_relative_localization_error: n_samples [{}] < 10".format(n_samples))
        return

    print_info("compute_relative_localization_error: computing re relations with {n} samples".format(n=n_samples))
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

    print_info("compute_relative_localization_error: computing metric re_t_unsorted")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=base_link_poses_file_path,
                     relations_path=relations_re_file_path,
                     weights="{1, 1, 1, 0, 0, 0}",
                     log_path=path.join(log_output_folder, "re_t.log"),
                     errors_path=path.join(results_output_folder, "re_t.csv"),
                     unsorted_errors_path=path.join(results_output_folder, "re_t_unsorted_errors"))

    print_info("compute_relative_localization_error: computing metric re_r_unsorted")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=base_link_poses_file_path,
                     relations_path=relations_re_file_path,
                     weights="{0, 0, 0, 1, 1, 1}",
                     log_path=path.join(log_output_folder, "re_r.log"),
                     errors_path=path.join(results_output_folder, "re_r.csv"),
                     unsorted_errors_path=path.join(results_output_folder, "re_r_unsorted_errors"))

    # ordered relations
    ordered_relations_file_path = path.join(results_output_folder, "ordered_relations")
    with open(ordered_relations_file_path, "w") as relations_file_ordered:

        idx_delta = len(ground_truth_acceptable_times)/n_samples
        if idx_delta == 0:
            print_error("compute_relative_localization_error: len(ground_truth_acceptable_times) [{l}] < n_samples [{n}]".format(l=len(ground_truth_acceptable_times), n=n_samples))
            idx_delta = 1

        print_info("compute_relative_localization_error: computing ordered relations with {n} samples".format(n=len(ground_truth_acceptable_times[0::idx_delta][0:-1])))

        for idx, first_stamp in enumerate(ground_truth_acceptable_times[0::idx_delta][0:-1]):
            second_stamp = ground_truth_acceptable_times[idx + idx_delta]

            first_pos = ground_truth_dict[first_stamp]
            second_pos = ground_truth_dict[second_stamp]

            rel = get_matrix_diff(first_pos, second_pos)
            x = rel[0, 3]
            y = rel[1, 3]
            theta = math.atan2(rel[1, 0], rel[0, 0])

            relations_file_ordered.write("{first_stamp} {second_stamp} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(first_stamp=first_stamp, second_stamp=second_stamp, x=x, y=y, theta=theta))

    print_info("compute_relative_localization_error: computing metric ordered_t")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=base_link_poses_file_path,
                     relations_path=ordered_relations_file_path,
                     weights="{1, 1, 1, 0, 0, 0}",
                     log_path=path.join(log_output_folder, "ordered_t.log"),
                     errors_path=path.join(results_output_folder, "ordered_t.csv"),
                     unsorted_errors_path=path.join(results_output_folder, "ordered_t_unsorted_errors"))

    print_info("compute_relative_localization_error: computing metric ordered_r")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=base_link_poses_file_path,
                     relations_path=ordered_relations_file_path,
                     weights="{0, 0, 0, 1, 1, 1}",
                     log_path=path.join(log_output_folder, "ordered_r.log"),
                     errors_path=path.join(results_output_folder, "ordered_r.csv"),
                     unsorted_errors_path=path.join(results_output_folder, "ordered_r_unsorted_errors"))


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


def compute_absolute_localization_error(results_output_folder, base_link_poses_file_path, ground_truth_file_path):

    # prepare folder structure
    if not path.exists(results_output_folder):
        os.makedirs(results_output_folder)

    # check required files exist
    if not path.isfile(base_link_poses_file_path):
        print_error("compute_relative_localization_error: base_link_poses_file_path file not found {}".format(base_link_poses_file_path))
        return

    if not path.isfile(ground_truth_file_path):
        print_error("compute_relative_localization_error: ground_truth_poses file not found {}".format(ground_truth_file_path))
        return

    # compute matching ground truth and slam poses times
    print_info("compute_absolute_localization_error: computing matching ground truth and slam poses times")
    ground_truth_dict = dict()
    with open(ground_truth_file_path, "r") as ground_truth_file:
        for line in ground_truth_file:
            time, x, y, theta = map(float, line.split(', '))
            ground_truth_dict[time] = (x, y, theta)

    base_link_dict = dict()
    with open(base_link_poses_file_path, "r") as base_link_file:
        for line in base_link_file:
            x, y, theta, time = map(float, line.split(' ')[-4:])
            base_link_dict[time] = (x, y, theta)

    matching_poses_dict = dict()
    for time in ground_truth_dict.keys():
        if time in base_link_dict:
            matching_poses_dict[time] = (base_link_dict[time], ground_truth_dict[time])

    print_info("compute_absolute_localization_error: found {} matching poses".format(len(matching_poses_dict)))

    def euclidean_distance(poses):
        a, b = poses
        a_x, a_y, _ = a
        b_x, b_y, _ = b

        return np.sqrt(np.sum((np.array((a_x, a_y)) - np.array((b_x, b_y)))**2))

    absolute_localization_error = sum(map(euclidean_distance, matching_poses_dict.values()))
    print(absolute_localization_error)

    result_file_path = path.join(results_output_folder, "absolute_localization_error")
    with open(result_file_path, "w") as result_file:
        result_file.write("{}\n".format(absolute_localization_error))


def compute_trajectory_length(results_output_folder, ground_truth_file_path):

    # prepare folder structure
    if not path.exists(results_output_folder):
        os.makedirs(results_output_folder)

    # check required files exist
    if not path.isfile(ground_truth_file_path):
        print_error("compute_trajectory_length: ground_truth_poses file not found {}".format(ground_truth_file_path))
        return

    # compute matching ground truth and slam poses times
    print_info("compute_trajectory_length: computing trajectory points list")
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

    print(trajectory_length)
    result_file_path = path.join(results_output_folder, "trajectory_length")
    with open(result_file_path, "w") as result_file:
        result_file.write("{}\n".format(trajectory_length))


def compute_localization_metrics(run_output_folder):
    """
    Given a run folder path, compute relation files and SLAM metric results
    """
    base_link_correction_poses_path = path.join(run_output_folder, "benchmark_data", "base_link_correction_poses")
    base_link_poses_path = path.join(run_output_folder, "benchmark_data", "base_link_poses")
    ground_truth_poses_path = path.join(run_output_folder, "benchmark_data", "ground_truth_poses")

    metric_results_path = path.join(run_output_folder, "metric_results")
    relative_localisation_correction_error_path = path.join(metric_results_path, "relative_localisation_correction_error")
    relative_localisation_error_path = path.join(metric_results_path, "relative_localisation_error")
    absolute_localisation_correction_error_path = path.join(metric_results_path, "absolute_localisation_correction_error")
    absolute_localisation_error_path = path.join(metric_results_path, "absolute_localisation_error")

    logs_folder_path = path.join(run_output_folder, "logs")

    print_info("compute_localization_metrics: compute_trajectory_length")
    compute_trajectory_length(metric_results_path, ground_truth_poses_path)

    print_info("compute_localization_metrics: compute_relative_localization_error on base_link_correction_poses")
    compute_relative_localization_error(relative_localisation_correction_error_path, logs_folder_path, base_link_correction_poses_path, ground_truth_poses_path)

    print_info("compute_localization_metrics: compute_relative_localization_error on base_link_poses_path")
    compute_relative_localization_error(relative_localisation_error_path, logs_folder_path, base_link_poses_path, ground_truth_poses_path)

    print_info("compute_localization_metrics: compute_absolute_localization_error on base_link_correction_poses")
    compute_absolute_localization_error(absolute_localisation_correction_error_path, base_link_correction_poses_path, ground_truth_poses_path)

    print_info("compute_localization_metrics: compute_absolute_localization_error on base_link_poses_path")
    compute_absolute_localization_error(absolute_localisation_error_path, base_link_poses_path, ground_truth_poses_path)


if __name__ == '__main__':
    run_folders = filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling_output/test_1/*")))
    # last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
    # print("last run folder:", last_run_folder)
    for run_folder in run_folders:
        print_info("main: compute_localization_metrics in {}".format(run_folder))
        compute_localization_metrics(path.expanduser(run_folder))
