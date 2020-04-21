#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import argparse
import os
import pickle
import sys
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
from os import path

import matplotlib.pyplot as plt

from performance_modelling_ros.utils import print_info, print_error


def cm_to_body_parts(*argv):
    inch = 2.54
    if isinstance(argv[0], tuple):
        return tuple(x / inch for x in argv[0])
    else:
        return tuple(x / inch for x in argv)


def get_simple_value(result_path):
    with open(result_path) as result_file:
        return result_file.read()


def get_csv(result_path):
    df = pd.read_csv(result_path, sep=', ', engine='python')
    return df


def get_yaml(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        return yaml.load(yaml_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the analysis of the benchmark results.')

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder containing the result the runs. Defaults to ~/ds/performance_modelling_output/test_1/',
                        type=str,
                        default="~/ds/performance_modelling_output/test_1/",
                        required=False)

    parser.add_argument('-o', dest='output_folder',
                        help='Folder in which the results will be placed. Defaults to ~/ds/performance_modelling_analysis/test_1/',
                        type=str,
                        default="~/ds/performance_modelling_analysis/test_1/",
                        required=False)

    parser.add_argument('-c', dest='cache_file',
                        help='If set the run data is cached and read from CACHE_FILE. CACHE_FILE defaults to ~/ds/performance_modelling_analysis_cache.pkl',
                        action='store_const',
                        const="~/ds/performance_modelling_analysis_cache.pkl",
                        default=None,
                        required=False)

    parser.add_argument('-i', dest='invalidate_cache',
                        help='If set invalidate the cached run data. If set the run data is re-read and the cache file is updated.',
                        action='store_true',
                        default=False,
                        required=False)

    parser.add_argument('--plot-everything', dest='plot_everything',
                        help='Plot everything.',
                        action='store_true',
                        default=False,
                        required=False)

    parser.add_argument('--plot-metrics-by-config', dest='plot_metrics_by_config',
                        help='Plot metrics by config.',
                        action='store_true',
                        default=False,
                        required=False)

    parser.add_argument('--plot-metrics-by-parameter', dest='plot_metrics_by_parameter',
                        help='Plot metrics by parameter using the specified aggregation function.',
                        action='store_true',
                        default=False,
                        required=False)

    parser.add_argument('--plot-metrics-by-metric', dest='plot_metrics_by_metric',
                        help='Plot metrics by metric.',
                        action='store_true',
                        default=False,
                        required=False)

    parser.add_argument('--plot-metric-histograms', dest='plot_metric_histograms',
                        help='Plot metric histograms.',
                        action='store_true',
                        default=False,
                        required=False)

    args = parser.parse_args()
    base_run_folder = path.expanduser(args.base_run_folder)
    output_folder = path.expanduser(args.output_folder)
    invalidate_cache = args.invalidate_cache
    if args.cache_file is not None:
        cache_file_path = path.expanduser(args.cache_file)
    else:
        cache_file_path = None
        if invalidate_cache:
            print_error("Flag invalidate_cache is set but no cache file is provided.")

    if not path.isdir(base_run_folder):
        print_error("base_run_folder does not exists or is not a directory".format(base_run_folder))
        sys.exit(-1)

    if not path.exists(output_folder):
        os.makedirs(output_folder)

    run_folders = filter(path.isdir, glob.glob(path.abspath(base_run_folder) + '/*'))
    print("base_run_folder:", base_run_folder)

    if not invalidate_cache and cache_file_path is not None and path.exists(cache_file_path):
        print_info("reading run data from cache")
        with open(cache_file_path) as f:
            cache = pickle.load(f)
        metrics_by_config = cache['metrics_by_config']
        metrics_by_run = cache['metrics_by_run']
        metric_names = cache['metric_names']
    else:
        metrics_by_config = defaultdict(lambda: defaultdict(list))
        metrics_by_run = list()
        metric_names = set()

        # collect results from each run
        print_info("reading run data")
        for i, run_folder in enumerate(run_folders):
            run_record = dict()
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

            if 'local_components_configuration' in run_info:
                gmapping_configuration_file_path = path.join(run_folder, run_info['local_components_configuration']['gmapping'])
            else:
                # if the local path is not in the run info, use the (deprecated) absolute path of the original configuration file
                gmapping_configuration_file_path = run_info['components_configuration']['gmapping']

            gmapping_configuration = get_yaml(gmapping_configuration_file_path)

            particles = gmapping_configuration['particles']
            delta = gmapping_configuration['delta']
            maxUrange = gmapping_configuration['maxUrange']
            environment_name = path.basename(run_info['environment_folder'])
            config = (particles, delta, maxUrange, environment_name)

            run_record['config'] = config
            run_record['failure'] = 0
            metric_names.add('failure')
            # TODO log each failure mode

            try:
                run_events = get_csv(path.join(benchmark_data_folder, "run_events.csv"))
                map_metrics = get_yaml(path.join(metric_results_folder, "map_metrics.yaml"))
                localisation_metrics = get_yaml(path.join(metric_results_folder, "localisation_metrics.yaml"))
                navigation_metrics = get_yaml(path.join(metric_results_folder, "navigation_metrics.yaml"))
                computation_metrics = get_yaml(path.join(metric_results_folder, "computation_metrics.yaml"))
            except IOError as e:
                metrics_by_config['failure'][config].append(1)
                run_record['failure'] = 1
                metrics_by_run.append(run_record)
                continue

            trajectory_length = localisation_metrics['trajectory_length']
            if trajectory_length < 3.0 or trajectory_length is None:
                metrics_by_config['failure'][config].append(1)
                run_record['failure'] = 1
                metrics_by_run.append(run_record)
                continue

            normalised_explored_area = map_metrics['explored_area']['normalised_explored_area']
            if normalised_explored_area < 0.1 or normalised_explored_area is None:
                metrics_by_config['failure'][config].append(1)
                run_record['failure'] = 1
                metrics_by_run.append(run_record)
                continue

            metrics_by_config['normalised_explored_area'][config].append(normalised_explored_area)

            metrics_by_config['mean_absolute_correction_error'][config].append(localisation_metrics['absolute_localization_error']['mean'])

            run_start_time = float(run_events[run_events["event"] == "run_start"]["timestamp"])
            supervisor_finish_time = float(run_events[run_events["event"] == "supervisor_finished"]["timestamp"])
            run_execution_time = supervisor_finish_time - run_start_time
            metrics_by_config['run_execution_time'][config].append(run_execution_time)
            run_record['run_execution_time'] = run_execution_time
            metric_names.add('run_execution_time')

            normalised_gmapping_cpu_time = computation_metrics['cpu_and_memory_usage']['slam_gmapping_accumulated_cpu_time'] / run_execution_time
            metrics_by_config['normalised_slam_cpu_time'][config].append(normalised_gmapping_cpu_time)
            run_record['normalised_slam_cpu_time'] = normalised_gmapping_cpu_time
            metric_names.add('normalised_slam_cpu_time')

            explored_area = map_metrics['explored_area']['result_map']['area']['free']
            normalised_gmapping_uss = computation_metrics['cpu_and_memory_usage']['slam_gmapping_uss'] / explored_area
            metrics_by_config['normalised_slam_memory'][config].append(normalised_gmapping_uss)
            run_record['normalised_slam_memory'] = normalised_gmapping_uss
            metric_names.add('normalised_slam_memory')

            metrics_by_config['failure'][config].append(0)
            metrics_by_run.append(run_record)
            del run_record

            print_info("reading run data: {}%".format((i + 1)*100/len(run_folders)), replace_previous_line=True)

        # save cache
        if cache_file_path is not None:
            metrics_by_config = dict(metrics_by_config)
            cache = {'metrics_by_config': metrics_by_config, 'metrics_by_run': metrics_by_run, 'metric_names': metric_names}
            with open(cache_file_path, 'w') as f:
                pickle.dump(cache, f)

    parameter_names = ('particles', 'delta', 'maxUrange', 'environment')
    configs_sets = defaultdict(set)
    metrics_x_y_by_config = set()
    for metric_name in metrics_by_config.keys():
        for config in metrics_by_config[metric_name].keys():
            metrics_x_y_by_config.add(config)

            particles, delta, maxUrange, environment = config
            configs_sets['particles'].add(particles)
            configs_sets['delta'].add(delta)
            configs_sets['maxUrange'].add(maxUrange)
            configs_sets['environment'].add(environment)

    # plot metrics grouped by configuration
    if args.plot_everything or args.plot_metrics_by_config:
        print_info("plot metrics grouped by configuration")
        metrics_by_config_folder = path.join(output_folder, "metrics_by_config")
        if not path.exists(metrics_by_config_folder):
            os.makedirs(metrics_by_config_folder)

        for metric_name in metrics_by_config.keys():
            fig, ax = plt.subplots()
            fig.set_size_inches(*cm_to_body_parts(30, 30))
            ax.margins(0.15)
            x_ticks = list()
            for i, (config, metric_values) in enumerate(metrics_by_config[metric_name].items()):
                ax.plot([i] * len(metric_values), metric_values, marker='_', linestyle='', ms=20, color='black')
                x_ticks.append(str(config))

            ax.set_title(metric_name)
            ax.set_xticks(range(len(x_ticks)))
            ax.set_xticklabels(x_ticks, fontdict={'rotation': 'vertical'})

            fig.savefig(path.join(metrics_by_config_folder, "{}.svg".format(metric_name)), bbox_inches='tight')
            plt.close(fig)

    # plot metrics in function of single configuration parameters
    if args.plot_everything or args.plot_metrics_by_parameter:
        aggregation_function = {'std': np.std, 'mean': np.mean, 'median': np.median, 'min': np.min, 'max': np.max}
        for aggregation_function_name in aggregation_function.keys():

            print_info("plot metrics by parameter using {a}".format(a=aggregation_function_name))
            metrics_by_parameter_folder = path.join(output_folder, "metrics_by_parameter_using_{a}".format(a=aggregation_function_name))
            if not path.exists(metrics_by_parameter_folder):
                os.makedirs(metrics_by_parameter_folder)

            for i, metric_name in enumerate(metrics_by_config.keys()):

                configs_df = pd.DataFrame.from_records(columns=parameter_names, data=list(set(metrics_by_config[metric_name].keys())))

                for parameter_name in parameter_names:
                    # plot lines for same-parameter metric values
                    fig, ax = plt.subplots()
                    fig.set_size_inches(*cm_to_body_parts(40, 40))
                    ax.margins(0.15)
                    ax.set_xlabel(parameter_name)
                    ax.set_ylabel(metric_name)

                    other_parameters = list(set(parameter_names) - {parameter_name})
                    grouped_parameter_values = configs_df.sort_values(by=other_parameters).groupby(other_parameters)
                    for p, configs_group in list(grouped_parameter_values):
                        sorted_configs_group = configs_group.sort_values(by=parameter_name)
                        other_parameter_values = next(sorted_configs_group[other_parameters].itertuples(index=False, name='config'))
                        parameter_values = sorted_configs_group[parameter_name]
                        metric_values = map(lambda c: aggregation_function[aggregation_function_name](metrics_by_config[metric_name][tuple(list(c))]), sorted_configs_group.itertuples(index=False))
                        ax.plot(parameter_values, metric_values, marker='o', ms=5, label=str(other_parameter_values))

                    ax.grid(color='black', alpha=0.5, linestyle='solid')
                    ax.legend()
                    fig.savefig(path.join(metrics_by_parameter_folder, "{}_by_{}_using_{}.svg".format(metric_name, parameter_name, aggregation_function_name)), bbox_inches='tight')
                    plt.close(fig)

                print_info("plot metrics by parameter using {a}: {p}%".format(a=aggregation_function_name, p=(i + 1)*100/len(metrics_by_config.keys())), replace_previous_line=True)

    # plot metrics in function of other metrics
    if args.plot_everything or args.plot_metrics_by_metric:
        print_info("plot metrics by metric")

        metrics_by_metric_folder = path.join(output_folder, "metrics_by_metric")
        if not path.exists(metrics_by_metric_folder):
            os.makedirs(metrics_by_metric_folder)

        progress_counter = 0
        progress_total = len(metric_names)**2 - len(metric_names)

        for metric_x_name in metric_names:
            for metric_y_name in metric_names - {metric_x_name}:

                fig, ax = plt.subplots()
                fig.set_size_inches(*cm_to_body_parts(40, 40))
                ax.margins(0.15)
                ax.set_xlabel(metric_x_name)
                ax.set_ylabel(metric_y_name)

                # get the configurations associated with metric x and y, grouped by config
                metrics_x_y_by_config = defaultdict(lambda: {'x': list(), 'y': list()})
                for run_record in metrics_by_run:
                    if metric_x_name in run_record and metric_y_name in run_record:
                        metrics_x_y_by_config[run_record['config']]['x'].append(run_record[metric_x_name])
                        metrics_x_y_by_config[run_record['config']]['y'].append(run_record[metric_y_name])

                # plot scatter graph for same-config metric x, y values (each config has a different color)
                for config, metric_values in metrics_x_y_by_config.items():
                    ax.scatter(metric_values['x'], metric_values['y'])

                ax.grid(color='black', alpha=0.5, linestyle='solid')

                fig_name = "{x}_to_{y}.svg".format(y=metric_y_name, x=metric_x_name)
                fig.savefig(path.join(metrics_by_metric_folder, fig_name), bbox_inches='tight')
                plt.close(fig)

                progress_counter += 1
                print_info("plot metrics by metric: {}%".format(progress_counter * 100 / progress_total), replace_previous_line=True)

    # plot metrics in function of other metrics
    if args.plot_everything or args.plot_metric_histograms:
        print_info("plot metric histograms")

        metric_histograms_folder = path.join(output_folder, "metric_histograms")
        if not path.exists(metric_histograms_folder):
            os.makedirs(metric_histograms_folder)

        progress_counter = 0
        progress_total = len(metric_names)

        for metric_x_name in metric_names:
            fig, ax = plt.subplots()
            fig.set_size_inches(*cm_to_body_parts(40, 40))
            ax.margins(0.15)
            ax.set_xlabel(metric_x_name)

            metric_values = list()
            for run_record in metrics_by_run:
                if metric_x_name in run_record:
                    metric_values.append(run_record[metric_x_name])

            # plot histogram graph for metric x values
            ax.hist(metric_values, bins=100)

            fig_name = "{}.svg".format(metric_x_name)
            fig.savefig(path.join(metric_histograms_folder, fig_name), bbox_inches='tight')
            plt.close(fig)

            progress_counter += 1
            print_info("plot metric histograms: {}%".format(progress_counter * 100 / progress_total), replace_previous_line=True)

    # TODO plot metric sensitivity by param
