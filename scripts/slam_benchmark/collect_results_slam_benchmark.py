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


def cm_to_stupid(*argv):
    inch = 2.54
    if isinstance(argv[0], tuple):
        return tuple(x / inch for x in argv[0])
    else:
        return tuple(x / inch for x in argv)


def get_simple_value(result_path):
    with open(result_path) as result_file:
        return result_file.read()


def get_metric_evaluator_mean(result_path):
    df = pd.read_csv(result_path, sep=', ', engine='python')
    return df['Mean'][0]


def get_yaml(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        return yaml.load(yaml_file)


def aggregation_function(metric_values_list):
    return np.mean(metric_values_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the benchmark')

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder containing the result the runs.',
                        type=str,
                        default="~/ds/performance_modelling_output/test_1/",
                        required=False)

    parser.add_argument('-o', dest='output_folder',
                        help='Folder in which the results will be placed.',
                        type=str,
                        default="~/ds/performance_modelling_analysis/test_1/",
                        required=False)

    parser.add_argument('-c', dest='cache_file',
                        help='Folder in which the cache will be placed.',
                        type=str,
                        default="~/ds/performance_modelling_analysis_cache",
                        required=False)

    args = parser.parse_args()
    base_run_folder = path.expanduser(args.base_run_folder)
    output_folder = path.expanduser(args.output_folder)
    cache_file_path = path.expanduser(args.cache_file)

    if not path.isdir(base_run_folder):
        print_error("base_run_folder does not exists or is not a directory".format(base_run_folder))
        sys.exit(-1)

    if not path.exists(output_folder):
        os.makedirs(output_folder)

    run_folders = filter(path.isdir, glob.glob(path.abspath(base_run_folder) + '/*'))
    print("base_run_folder:", base_run_folder)

    if path.exists(cache_file_path):
        print_info("reading run data from cache")
        with open(cache_file_path) as f:
            cache = pickle.load(f)
        metrics_by_config = cache['metrics_by_config']
    else:
        metrics_by_config = defaultdict(lambda: defaultdict(list))

        # collect results from each run
        print_info("reading run data")
        for i, run_folder in enumerate(run_folders):
            metric_results_folder = path.join(run_folder, "metric_results")
            run_info_file_path = path.join(run_folder, "run_info.yaml")
            if not path.exists(metric_results_folder):
                print_error("metric_results_folder does not exists [{}]".format(metric_results_folder))
                continue
            if not path.exists(run_info_file_path):
                print_error("run_info file does not exists [{}]".format(run_info_file_path))
                continue

            run_info = get_yaml(run_info_file_path)

            gmapping_configuration_file_path = run_info['components_configuration']['gmapping']
            gmapping_configuration = get_yaml(gmapping_configuration_file_path)

            particles = gmapping_configuration['particles']
            delta = gmapping_configuration['delta']
            maxUrange = gmapping_configuration['maxUrange']
            environment_name = path.basename(run_info['environment_folder'])
            config = (particles, delta, maxUrange, environment_name)

            metrics_by_config['normalised_explored_area'][config].append(get_yaml(path.join(metric_results_folder, "normalised_explored_area.yaml"))['normalised_explored_area'])
            metrics_by_config['explored_area'][config].append(get_yaml(path.join(metric_results_folder, "normalised_explored_area.yaml"))['result_map']['area']['free'])

            trajectory_length = float(get_simple_value(path.join(metric_results_folder, "trajectory_length")))
            metrics_by_config['trajectory_length'][config].append(trajectory_length)

            metrics_by_config['normalised_absolute_error'][config].append(float(get_simple_value(path.join(metric_results_folder, "absolute_localisation_error", "absolute_localization_error"))) / trajectory_length)
            metrics_by_config['normalised_absolute_correction_error'][config].append(float(get_simple_value(path.join(metric_results_folder, "absolute_localisation_correction_error", "absolute_localization_error"))) / trajectory_length)

            # metrics_by_config['absolute_error'][config].append(float(get_simple_value(path.join(metric_results_folder, "absolute_localisation_error", "absolute_localization_error"))))
            # metrics_by_config['absolute_correction_error'][config].append(float(get_simple_value(path.join(metric_results_folder, "absolute_localisation_correction_error", "absolute_localization_error"))))

            ordered_r_path = path.join(metric_results_folder, "base_link_poses", "ordered_r.csv")
            ordered_t_path = path.join(metric_results_folder, "base_link_poses", "ordered_t.csv")
            re_r_path = path.join(metric_results_folder, "base_link_poses", "re_r.csv")
            re_t_path = path.join(metric_results_folder, "base_link_poses", "re_t.csv")

            correction_ordered_r_path = path.join(metric_results_folder, "base_link_correction_poses", "ordered_r.csv")
            correction_ordered_t_path = path.join(metric_results_folder, "base_link_correction_poses", "ordered_t.csv")
            correction_re_r_path = path.join(metric_results_folder, "base_link_correction_poses", "re_r.csv")
            correction_re_t_path = path.join(metric_results_folder, "base_link_correction_poses", "re_t.csv")

            if path.exists(ordered_r_path):
                metrics_by_config['normalised_ordered_r'][config].append(get_metric_evaluator_mean(ordered_r_path) / trajectory_length)
                metrics_by_config['ordered_r'][config].append(get_metric_evaluator_mean(ordered_r_path))

            if path.exists(ordered_t_path):
                metrics_by_config['normalised_ordered_t'][config].append(get_metric_evaluator_mean(ordered_t_path) / trajectory_length)
                metrics_by_config['ordered_t'][config].append(get_metric_evaluator_mean(ordered_t_path))

            if path.exists(re_r_path):
                metrics_by_config['normalised_re_r'][config].append(get_metric_evaluator_mean(re_r_path) / trajectory_length)
                metrics_by_config['re_r'][config].append(get_metric_evaluator_mean(re_r_path))

            if path.exists(re_t_path):
                metrics_by_config['normalised_re_t'][config].append(get_metric_evaluator_mean(re_t_path) / trajectory_length)
                metrics_by_config['re_t'][config].append(get_metric_evaluator_mean(re_t_path))

            if path.exists(correction_ordered_r_path):
                metrics_by_config['normalised_correction_ordered_r'][config].append(get_metric_evaluator_mean(correction_ordered_r_path) / trajectory_length)
                metrics_by_config['correction_ordered_r'][config].append(get_metric_evaluator_mean(correction_ordered_r_path))

            if path.exists(correction_ordered_t_path):
                metrics_by_config['normalised_correction_ordered_t'][config].append(get_metric_evaluator_mean(correction_ordered_t_path) / trajectory_length)
                metrics_by_config['correction_ordered_t'][config].append(get_metric_evaluator_mean(correction_ordered_t_path))

            if path.exists(correction_re_r_path):
                metrics_by_config['normalised_correction_re_r'][config].append(get_metric_evaluator_mean(correction_re_r_path) / trajectory_length)
                metrics_by_config['correction_re_r'][config].append(get_metric_evaluator_mean(correction_re_r_path))

            if path.exists(correction_re_t_path):
                metrics_by_config['normalised_correction_re_t'][config].append(get_metric_evaluator_mean(correction_re_t_path) / trajectory_length)
                metrics_by_config['correction_re_t'][config].append(get_metric_evaluator_mean(correction_re_t_path))

            print_info("reading run data: {}%".format((i + 1)*100/len(run_folders)), replace_previous_line=True)

        # save cache
        metrics_by_config = dict(metrics_by_config)
        cache = {'metrics_by_config': metrics_by_config}
        with open(cache_file_path, 'w') as f:
            pickle.dump(cache, f)

    parameter_names = ('particles', 'delta', 'maxUrange', 'environment')
    configs_sets = defaultdict(set)
    configs = set()
    for metric_name in metrics_by_config.keys():
        for config in metrics_by_config[metric_name].keys():
            configs.add(config)

            particles, delta, maxUrange, environment = config
            configs_sets['particles'].add(particles)
            configs_sets['delta'].add(delta)
            configs_sets['maxUrange'].add(maxUrange)
            configs_sets['environment'].add(environment)

    # plot metrics grouped by configuration
    plot_metrics_by_config = False
    if plot_metrics_by_config:
        print_info("plot metrics grouped by configuration")
        metrics_by_config_folder = path.join(output_folder, "metrics_by_config")
        if not path.exists(metrics_by_config_folder):
            os.makedirs(metrics_by_config_folder)

        for metric_name in metrics_by_config.keys():
            fig, ax = plt.subplots()
            fig.set_size_inches(*cm_to_stupid(30, 30))
            ax.margins(0.15)
            x_ticks = list()
            for i, (config, metric_values) in enumerate(metrics_by_config[metric_name].items()):
                ax.plot([i] * len(metric_values[metric_name]), metric_values[metric_name], marker='_', linestyle='', ms=20, color='black')
                x_ticks.append(str(config))

            ax.set_title(metric_name)
            ax.set_xticks(range(len(x_ticks)))
            ax.set_xticklabels(x_ticks, fontdict={'rotation': 'vertical'})

            fig.savefig(path.join(metrics_by_config_folder, "{}.svg".format(metric_name)), bbox_inches='tight')
            plt.close(fig)

    # plot metrics in function of single configuration parameters
    plot_metrics_by_parameter = True
    if plot_metrics_by_parameter:
        print_info("plot metrics by parameter")

        metrics_by_parameter_folder = path.join(output_folder, "metrics_by_parameter")
        if not path.exists(metrics_by_parameter_folder):
            os.makedirs(metrics_by_parameter_folder)

        for i, metric_name in enumerate(metrics_by_config.keys()):

            configs_df = pd.DataFrame.from_records(columns=parameter_names, data=list(set(metrics_by_config[metric_name].keys())))

            for parameter_name in parameter_names:
                # plot lines for same-parameter metric values
                fig, ax = plt.subplots()
                fig.set_size_inches(*cm_to_stupid(20, 20))
                ax.margins(0.15)
                ax.set_xlabel(parameter_name)
                ax.set_ylabel(metric_name)

                other_parameters = list(set(parameter_names) - {parameter_name})
                grouped_parameter_values = configs_df.sort_values(by=other_parameters).groupby(other_parameters)
                for p, configs_group in list(grouped_parameter_values):
                    sorted_configs_group = configs_group.sort_values(by=parameter_name)
                    other_parameter_values = next(sorted_configs_group[other_parameters].itertuples(index=False, name='config'))
                    parameter_values = sorted_configs_group[parameter_name]
                    # if len(parameter_values) < 3:
                    #     continue
                    metric_values = map(lambda c: aggregation_function(metrics_by_config[metric_name][tuple(list(c))]), sorted_configs_group.itertuples(index=False))
                    ax.plot(parameter_values, metric_values, marker='o', ms=5, label=str(other_parameter_values))

                ax.legend()
                fig.savefig(path.join(metrics_by_parameter_folder, "{}_by_{}.svg".format(metric_name, parameter_name)), bbox_inches='tight')
                plt.close(fig)

            print_info("plot metrics by parameter: {}%".format((i + 1)*100/len(metrics_by_config.keys())), replace_previous_line=True)
