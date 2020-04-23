#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
import yaml
from collections import defaultdict
from os import path
import pickle
from performance_modelling_py.utils import print_info, print_error


def cpu_and_memory_usage_metrics(ps_snapshots_folder_path):
    cpu_and_memory_usage_dict = dict()
    cpu_and_memory_usage_list_dict = defaultdict(list)

    # get list of all ps snapshots
    ps_snapshot_paths_list = sorted(glob.glob(path.join(ps_snapshots_folder_path, "ps_*.pkl")))

    for ps_snapshot_path in ps_snapshot_paths_list:
        try:
            with open(ps_snapshot_path) as ps_snapshot_file:
                ps_snapshot = pickle.load(ps_snapshot_file)
        except EOFError as e:
            print_error("Could not load pickled ps snapshot. Error: {t} {e}. Pickle file: {f}".format(e=e, t=type(e), f=ps_snapshot_path))
            continue

        for process_info in ps_snapshot:
            process_name = process_info['name']

            cpu_and_memory_usage_list_dict["{p}_uss".format(p=process_name)].append(process_info['memory_full_info'].uss)
            cpu_and_memory_usage_list_dict["{p}_rss".format(p=process_name)].append(process_info['memory_full_info'].rss)
            cpu_and_memory_usage_list_dict["{p}_accumulated_cpu_time".format(p=process_name)].append(process_info['cpu_times'].user + process_info['cpu_times'].system)

    for metric_name, metric_values in cpu_and_memory_usage_list_dict.items():
        cpu_and_memory_usage_dict[metric_name] = max(cpu_and_memory_usage_list_dict[metric_name])

    return cpu_and_memory_usage_dict


def compute_computation_metrics(run_output_folder):
    """
    Given a run folder path, compute computation metrics such as memory and CPU usage
    """
    ps_snapshots_folder_path = path.join(run_output_folder, "benchmark_data", "ps_snapshots")
    metric_results_folder_path = path.join(run_output_folder, "metric_results")
    metrics_result_file_path = path.join(metric_results_folder_path, "computation_metrics.yaml")

    metrics_result_dict = dict()
    metrics_result_dict['cpu_and_memory_usage'] = cpu_and_memory_usage_metrics(ps_snapshots_folder_path)

    if not path.exists(metric_results_folder_path):
        os.makedirs(metric_results_folder_path)

    with open(metrics_result_file_path, 'w') as metrics_result_file:
        yaml.dump(metrics_result_dict, metrics_result_file, default_flow_style=False)


if __name__ == '__main__':
    run_folders = filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling_output/test_1/*")))
    # last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
    # print("last run folder:", last_run_folder)
    for progress, run_folder in enumerate(run_folders):
        print_info("main: compute_computation_metrics {}% {}".format((progress + 1)*100/len(run_folders), run_folder))
        compute_computation_metrics(path.expanduser(run_folder))
