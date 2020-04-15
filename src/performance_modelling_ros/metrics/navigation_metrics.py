#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import yaml
from os import path
import numpy as np

from performance_modelling_ros.utils import print_info, print_error


def compute_cmd_vel_metrics(results_output_folder, cmd_vel_twists_file_path):

    sum_linear_cmd = 0.0
    sum_angular_cmd = 0.0
    sum_combined_cmd = 0.0

    with open(cmd_vel_twists_file_path, 'r') as cmd_vel_twists_file:
        for line in cmd_vel_twists_file.readlines():
            _, v_x, v_y, v_theta = map(float, line.split(', '))
            linear_cmd = np.abs(v_x) + np.abs(v_y)
            angular_cmd = np.abs(v_theta)

            sum_linear_cmd += linear_cmd
            sum_angular_cmd += angular_cmd
            sum_combined_cmd += linear_cmd * angular_cmd

    result_file_path = path.join(results_output_folder, "cmd_vel_metrics.yaml")
    with open(result_file_path, "w") as result_file:
        print("\n")
        print(result_file_path)
        print({'sum_linear_cmd': sum_linear_cmd,
               'sum_angular_cmd': sum_angular_cmd,
               'sum_combined_cmd': sum_combined_cmd})
        yaml.dump({'sum_linear_cmd': sum_linear_cmd,
                   'sum_angular_cmd': sum_angular_cmd,
                   'sum_combined_cmd': sum_combined_cmd},
                  result_file, default_flow_style=False)


def compute_navigation_metrics(run_output_folder):
    """
    Given a run folder path, compute navigation metric results
    """
    cmd_vel_twists_path = path.join(run_output_folder, "benchmark_data", "cmd_vel_twists")

    metric_results_path = path.join(run_output_folder, "metric_results")

    print_info("compute_navigation_metrics: compute_cmd_vel_metrics")
    compute_cmd_vel_metrics(metric_results_path, cmd_vel_twists_path)


if __name__ == '__main__':
    run_folders = filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling_output/test_1/*")))
    # last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
    # print("last run folder:", last_run_folder)
    for run_folder in run_folders:
        print_info("main: compute_localization_metrics in {}".format(run_folder))
        compute_navigation_metrics(path.expanduser(run_folder))
