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
            with open(ps_snapshot_path, 'rb') as ps_snapshot_file:
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
