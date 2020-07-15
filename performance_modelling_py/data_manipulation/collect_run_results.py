#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import argparse
import pickle
import sys
import yaml
import pandas as pd
from os import path

print("python version:", sys.version_info)

from performance_modelling_py.utils import print_info, print_error


def cm_to_body_parts(*argv):
    inch = 2.54
    if isinstance(argv[0], tuple):
        return tuple(x_cm / inch for x_cm in argv[0])
    else:
        return tuple(x_cm / inch for x_cm in argv)


def get_simple_value(result_path):
    with open(result_path) as result_file:
        return result_file.read()


def get_csv(result_path):
    df_csv = pd.read_csv(result_path, sep=', ', engine='python')
    return df_csv


def get_yaml(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        return yaml.load(yaml_file)


def collect_data(base_run_folder_path, cache_file_path=None, invalidate_cache=True):

    print(base_run_folder_path, cache_file_path, invalidate_cache)

    base_run_folder = path.expanduser(base_run_folder_path)
    if cache_file_path is not None:
        cache_file_path = path.expanduser(cache_file_path)
    elif invalidate_cache:
        print_error("Flag invalidate_cache is set but no cache file is provided.")

    if not path.isdir(base_run_folder):
        print_error("base_run_folder does not exists or is not a directory".format(base_run_folder))
        sys.exit(-1)

    print("base_run_folder:", base_run_folder)
    run_folders = sorted(list(filter(path.isdir, glob.glob(path.abspath(base_run_folder) + '/*'))))
    print(run_folders)

    if not invalidate_cache and cache_file_path is not None and path.exists(cache_file_path):
        print_info("reading run data from cache")
        with open(cache_file_path, 'rb') as f:
            cache = pickle.load(f)
        df = cache['df']
        parameter_names = cache['parameter_names']
    else:
        df = pd.DataFrame()
        parameter_names = set()

        # collect results from each run
        print_info("reading run data")
        for i, run_folder in enumerate(run_folders):
            metric_results_folder = path.join(run_folder, "metric_results")
            benchmark_data_folder = path.join(run_folder, "benchmark_data")
            run_info_file_path = path.join(run_folder, "run_info.yaml")

            if not path.exists(metric_results_folder):
                print_error("metric_results_folder does not exists [{}]".format(metric_results_folder))
                continue
            if not path.exists(run_info_file_path):
                print_error("run_info file does not exists [{}]".format(run_info_file_path))
                continue

            run_info = get_yaml(run_info_file_path)

            run_record = dict()

            for parameter_name, parameter_value in run_info['run_parameters'].items():
                parameter_names.add(parameter_name)
                run_record[parameter_name] = parameter_value

            run_record['environment_name'] = path.basename(run_info['environment_folder'])
            run_record['run_folder'] = path.basename(run_folder)

            run_record['failure_rate'] = 0

            try:
                run_events = get_csv(path.join(benchmark_data_folder, "run_events.csv"))
                metrics_dict = get_yaml(path.join(metric_results_folder, "metrics.yaml"))
            except IOError as e:
                run_record['failure_rate'] = 1
                df = df.append(run_record, ignore_index=True)
                continue

            trajectory_length = metrics_dict['trajectory_length']
            if trajectory_length < 3.0 or trajectory_length is None:
                run_record['failure_rate'] = 1
                df = df.append(run_record, ignore_index=True)
                continue

            if metrics_dict['absolute_localization_correction_error'] is not None:
                run_record['mean_absolute_correction_error'] = metrics_dict['absolute_localization_correction_error']['mean']

            if metrics_dict['absolute_localization_error'] is not None:
                run_record['mean_absolute_error'] = metrics_dict['absolute_localization_error']['mean']

            run_start_events = run_events["event"] == "run_start"
            run_completed_events = run_events["event"] == "run_completed"
            if len(run_start_events) == 0 or len(run_completed_events) == 0:
                run_record['failure_rate'] = 1
                df = df.append(run_record, ignore_index=True)
                continue

            run_start_time = float(run_events[run_events["event"] == "run_start"]["timestamp"])
            supervisor_finish_time = float(run_events[run_events["event"] == "run_completed"]["timestamp"])
            run_execution_time = supervisor_finish_time - run_start_time
            run_record['run_execution_time'] = run_execution_time
            
            if metrics_dict['cpu_and_memory_usage'] is not None and 'amcl_accumulated_cpu_time' in metrics_dict['cpu_and_memory_usage']:
                run_record['normalised_cpu_time'] = metrics_dict['cpu_and_memory_usage']['amcl_accumulated_cpu_time'] / run_execution_time

            if metrics_dict['cpu_and_memory_usage'] is not None and 'amcl_uss' in metrics_dict['cpu_and_memory_usage']:
                run_record['max_memory'] = metrics_dict['cpu_and_memory_usage']['amcl_uss']

            if metrics_dict['cpu_and_memory_usage'] is not None and 'localization_slam_toolbox_node_accumulated_cpu_time' in metrics_dict['cpu_and_memory_usage']:
                run_record['normalised_cpu_time'] = metrics_dict['cpu_and_memory_usage']['localization_slam_toolbox_node_accumulated_cpu_time'] / run_execution_time

            if metrics_dict['cpu_and_memory_usage'] is not None and 'localization_slam_toolbox_node_uss' in metrics_dict['cpu_and_memory_usage']:
                run_record['max_memory'] = metrics_dict['cpu_and_memory_usage']['localization_slam_toolbox_node_uss']

            df = df.append(run_record, ignore_index=True)

            print_info("reading run data: {}%".format((i + 1)*100/len(run_folders)), replace_previous_line=True)

        # save cache
        if cache_file_path is not None:
            cache = {'df': df, 'parameter_names': parameter_names}
            with open(cache_file_path, 'wb') as f:
                pickle.dump(cache, f)

    metric_names = set(df.columns) - parameter_names
    
    pd.options.display.width = 204
    print("parameter_names", parameter_names)
    print("metric_names", metric_names)
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the analysis of the benchmark results.')

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder containing the result the runs. Defaults to ~/ds/performance_modelling_output/test_1/',
                        type=str,
                        default="~/ds/performance_modelling/output/test_localization",
                        required=False)

    parser.add_argument('-c', dest='cache_file',
                        help='If set the run data is cached and read from CACHE_FILE. CACHE_FILE defaults to ~/ds/performance_modelling_analysis_cache.pkl',
                        default="~/ds/performance_modelling/output/test_localization_ache.pkl",
                        required=False)

    parser.add_argument('-i', dest='invalidate_cache',
                        help='If set invalidate the cached run data. If set the run data is re-read and the cache file is updated.',
                        action='store_true',
                        default=False,
                        required=False)

    args = parser.parse_args()
    collect_data(args.base_run_folder, args.cache_file, True)
