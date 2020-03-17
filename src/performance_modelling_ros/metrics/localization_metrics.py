#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import numpy as np
import os
import random
from os import path
from scipy.stats import t
import subprocess

import roslib
import roslib.packages

from performance_modelling_ros.utils import print_info


def metric_evaluator(exec_path, poses_path, relations_path, weights, log_path, errors_path, unsorted_errors_path=None):
    with open(log_path, 'w') as stdout_log_file:
        if unsorted_errors_path is None:
            p = subprocess.Popen([exec_path, "-s", poses_path, "-r", relations_path, "-w", weights, "-e", errors_path], stdout=stdout_log_file)
        else:
            p = subprocess.Popen([exec_path, "-s", poses_path, "-r", relations_path, "-w", weights, "-e", errors_path, "-eu", unsorted_errors_path], stdout=stdout_log_file)
        p.wait()


def compute_relations_and_metrics(run_output_folder, results_output_folder, log_output_folder, base_link_poses_file_path, ground_truth_file_path, metric_evaluator_exec_path, alpha=0.99, max_error=0.02):
    """
    Generates the ordered and the random relations files
    """

    ground_truth_dict = dict()
    with open(ground_truth_file_path, "r") as ground_truth_file:
        for line in ground_truth_file:
            time, x, y, theta = map(float, line.split(', '))
            ground_truth_dict[time] = (x, y, theta)

    # random relations
    print_info("computing random relations")
    relations_re_file_path = path.join(run_output_folder, "re_relations")
    with open(relations_re_file_path, "w") as relations_file_re:

        if len(ground_truth_dict.keys()) == 0:
            return

        n_samples = 500
        for _ in range(n_samples):
            first_stamp = float(random.choice(ground_truth_dict.keys()))
            second_stamp = float(random.choice(ground_truth_dict.keys()))
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
    print_info("computing metric summary_t")
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
    print_info("computing metric summary_r")
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

    print_info("computing re relations with {n} samples".format(n=n_samples))
    with open(relations_re_file_path, "w") as relations_file_re:
        for _ in range(n_samples):
            first_stamp = float(random.choice(ground_truth_dict.keys()))
            second_stamp = float(random.choice(ground_truth_dict.keys()))
            if first_stamp > second_stamp:
                first_stamp, second_stamp = second_stamp, first_stamp
            first_pos = ground_truth_dict[first_stamp]
            second_pos = ground_truth_dict[second_stamp]

            rel = get_matrix_diff(first_pos, second_pos)
            x = rel[0, 3]
            y = rel[1, 3]
            theta = math.atan2(rel[1, 0], rel[0, 0])

            relations_file_re.write("{first_stamp} {second_stamp} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(first_stamp=first_stamp, second_stamp=second_stamp, x=x, y=y, theta=theta))

    print_info("computing metric re_t_unsorted")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=base_link_poses_file_path,
                     relations_path=relations_re_file_path,
                     weights="{1, 1, 1, 0, 0, 0}",
                     log_path=path.join(log_output_folder, "re_t.log"),
                     errors_path=path.join(results_output_folder, "re_t.csv"),
                     unsorted_errors_path=path.join(results_output_folder, "re_t_unsorted_errors"))

    print_info("computing metric re_r_unsorted")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=base_link_poses_file_path,
                     relations_path=relations_re_file_path,
                     weights="{0, 0, 0, 1, 1, 1}",
                     log_path=path.join(log_output_folder, "re_r.log"),
                     errors_path=path.join(results_output_folder, "re_r.csv"),
                     unsorted_errors_path=path.join(results_output_folder, "re_r_unsorted_errors"))

    # ordered relations
    ordered_relations_file_path = path.join(run_output_folder, "ordered_relations")
    relations_file_ordered = open(ordered_relations_file_path, "w")
    ground_truth_sorted_indices = sorted(ground_truth_dict)
    print_info("computing ordered relations with {n} samples".format(n=len(ground_truth_sorted_indices)/10))

    idx = 1
    idx_delta = 10
    first_stamp = ground_truth_sorted_indices[idx]
    while idx + idx_delta < len(ground_truth_sorted_indices):
        second_stamp = ground_truth_sorted_indices[idx + idx_delta]
        if first_stamp in ground_truth_dict.keys():
            first_pos = ground_truth_dict[first_stamp]
            if second_stamp in ground_truth_dict.keys():
                second_pos = ground_truth_dict[second_stamp]
                rel = get_matrix_diff(first_pos, second_pos)
                x = rel[0, 3]
                y = rel[1, 3]
                theta = math.atan2(rel[1, 0], rel[0, 0])

                relations_file_ordered.write("{first_stamp} {second_stamp} {x} {y} 0.000000 0.000000 0.000000 {theta}\n".format(first_stamp=first_stamp, second_stamp=second_stamp, x=x, y=y, theta=theta))

        first_stamp = second_stamp
        idx += idx_delta

    relations_file_ordered.close()

    print_info("computing metric ordered_t")
    metric_evaluator(exec_path=metric_evaluator_exec_path,
                     poses_path=base_link_poses_file_path,
                     relations_path=ordered_relations_file_path,
                     weights="{1, 1, 1, 0, 0, 0}",
                     log_path=path.join(log_output_folder, "ordered_t.log"),
                     errors_path=path.join(results_output_folder, "ordered_t.csv"),
                     unsorted_errors_path=path.join(results_output_folder, "ordered_t_unsorted_errors"))

    print_info("computing metric ordered_r")
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

    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    theta1 = p1[2]
    theta2 = p2[2]

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


def compute_localization_metrics(run_output_folder):
    """
    Given a run folder path, compute relation files and SLAM metric results
    """
    base_link_poses_path = path.join(run_output_folder, "base_link_poses")
    ground_truth_poses_path = path.join(run_output_folder, "ground_truth_poses")
    metric_results_path = path.join(run_output_folder, "metric_results")
    log_files_path = path.join(run_output_folder, "logs")

    # create folders structure
    if not path.exists(metric_results_path):
        os.makedirs(metric_results_path)

    if not path.exists(log_files_path):
        os.makedirs(log_files_path)

    # check required files exist
    if not path.isfile(base_link_poses_path):
        print("compute_localization_metrics: base_link_poses file not found {}".format(base_link_poses_path))
        return

    if not path.isfile(ground_truth_poses_path):
        print("compute_localization_metrics: ground_truth_poses file not found {}".format(ground_truth_poses_path))
        return

    # find the metricEvaluator executable
    metric_evaluator_package_name = 'performance_modelling'
    metric_evaluator_exec_name = 'metricEvaluator'
    metric_evaluator_resources_list = roslib.packages.find_resource(metric_evaluator_package_name, metric_evaluator_exec_name)
    if len(metric_evaluator_resources_list) > 1:
        print("Multiple files named [{resource_name}}] in package [{package_name}]:%s".format(resource_name=metric_evaluator_exec_name, package_name=metric_evaluator_package_name))
        return
    elif len(metric_evaluator_resources_list) == 0:
        print("No files named [{resource_name}}] in package [{package_name}]:%s".format(resource_name=metric_evaluator_exec_name, package_name=metric_evaluator_package_name))
        return
    metric_evaluator_exec_path = metric_evaluator_resources_list[0]

    compute_relations_and_metrics(run_output_folder, metric_results_path, log_files_path, base_link_poses_path, ground_truth_poses_path, metric_evaluator_exec_path)
