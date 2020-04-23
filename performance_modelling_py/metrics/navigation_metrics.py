#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
import yaml
from os import path
import numpy as np

from performance_modelling_py.utils import print_info


def cmd_vel_metrics(cmd_vel_twists_file_path):

    cmd_vel_metrics_dict = dict()

    sum_linear_cmd = 0.0
    sum_angular_cmd = 0.0
    sum_combined_cmd = 0.0
    n = 0

    with open(cmd_vel_twists_file_path, 'r') as cmd_vel_twists_file:
        for line in cmd_vel_twists_file.readlines():
            _, v_x, v_y, v_theta = map(float, line.split(', '))
            linear_cmd = np.abs(v_x) + np.abs(v_y)
            angular_cmd = np.abs(v_theta)
            n += 1

            sum_linear_cmd += linear_cmd
            sum_angular_cmd += angular_cmd
            sum_combined_cmd += linear_cmd * angular_cmd

    cmd_vel_metrics_dict['sum_linear_cmd'] = float(sum_linear_cmd)
    cmd_vel_metrics_dict['sum_angular_cmd'] = float(sum_angular_cmd)
    cmd_vel_metrics_dict['sum_combined_cmd'] = float(sum_combined_cmd)

    cmd_vel_metrics_dict['mean_linear_cmd'] = float(sum_linear_cmd)/n
    cmd_vel_metrics_dict['mean_angular_cmd'] = float(sum_angular_cmd)/n
    cmd_vel_metrics_dict['mean_combined_cmd'] = float(sum_combined_cmd)/n

    return cmd_vel_metrics_dict


def compute_navigation_metrics(run_output_folder):
    """
    Given a run folder path, compute navigation metric results
    """
    cmd_vel_twists_path = path.join(run_output_folder, "benchmark_data", "cmd_vel_twists")

    metrics_result_folder_path = path.join(run_output_folder, "metric_results")
    metrics_result_file_path = path.join(metrics_result_folder_path, "navigation_metrics.yaml")

    metrics_result_dict = dict()
    metrics_result_dict['cmd_vel'] = cmd_vel_metrics(cmd_vel_twists_path)

    if not path.exists(metrics_result_folder_path):
        os.makedirs(metrics_result_folder_path)

    with open(metrics_result_file_path, 'w') as metrics_result_file:
        yaml.dump(metrics_result_dict, metrics_result_file, default_flow_style=False)


if __name__ == '__main__':
    run_folders = filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling_output/test_1/*")))
    # last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
    # print("last run folder:", last_run_folder)
    for progress, run_folder in enumerate(run_folders):
        print_info("main: compute_navigation_metrics {}% {}".format((progress + 1)*100/len(run_folders), run_folder))
        compute_navigation_metrics(path.expanduser(run_folder))
