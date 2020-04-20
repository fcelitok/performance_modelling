#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import numpy as np
import os
import yaml
from collections import defaultdict
from os import path
import pickle
from performance_modelling_ros.utils import print_info, print_error, backup_file_if_exists


def compute_computation_metrics(run_output_folder):
    """
    Given a run folder path, compute computation metrics such as memory and CPU usage
    """
    ps_snapshots_folder_path = path.join(run_output_folder, "benchmark_data", "ps_snapshots")
    metric_results_folder_path = path.join(run_output_folder, "metric_results")
    metric_result_file_path = path.join(metric_results_folder_path, "computation_metrics.yaml")

    if not path.exists(metric_results_folder_path):
        os.makedirs(metric_results_folder_path)

    # get list of all ps snapshots
    ps_snapshot_paths_list = sorted(glob.glob(path.join(ps_snapshots_folder_path, "ps_*.pkl")))

    metrics_dict = defaultdict(list)
    for ps_snapshot_path in ps_snapshot_paths_list:
        try:
            with open(ps_snapshot_path) as ps_snapshot_file:
                ps_snapshot = pickle.load(ps_snapshot_file)
        except EOFError as e:
            print_error("Could not load pickled ps snapshot. Error: {t} {e}. Pickle file: {f}".format(e=e, t=type(e), f=ps_snapshot_path))
            continue

        for process_info in ps_snapshot:
            process_name = process_info['name']

            metrics_dict["{p}_uss".format(p=process_name)].append(process_info['memory_full_info'].uss)
            metrics_dict["{p}_rss".format(p=process_name)].append(process_info['memory_full_info'].rss)
            metrics_dict["{p}_accumulated_cpu_time".format(p=process_name)].append(process_info['cpu_times'].user + process_info['cpu_times'].system)

    for metric_name, metric_values in metrics_dict.items():
        metrics_dict["{}_max".format(metric_name)] = max(metrics_dict[metric_name])

    with open(metric_result_file_path, 'w') as metric_result_file:
        yaml.dump(dict(metrics_dict), metric_result_file, default_flow_style=False)


if __name__ == '__main__':
    run_folders = filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling_output/test_1/*")))
    # last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
    # print("last run folder:", last_run_folder)
    # compute_computation_metrics(path.expanduser(last_run_folder))
    for run_folder in run_folders:
        print_info("main: compute_computation_metrics in {}".format(run_folder))
        compute_computation_metrics(path.expanduser(run_folder))
