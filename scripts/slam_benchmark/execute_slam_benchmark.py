#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import rospkg
import argparse
import os
import sys
import traceback
import yaml
from os import path
import itertools
import roslaunch

from performance_modelling_ros.utils import print_info, print_error
from slam_benchmark_supervisor_ros.slam_benchmark_run import BenchmarkRun

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the benchmark')

    parser.add_argument('-e', dest='environment_dataset_folder',
                        help='Dataset folder containg the stage environment.world file (recursively).',
                        type=str,
                        default="~/ds/performance_modelling_few_datasets/",
                        required=False)

    parser.add_argument('-c', dest='benchmark_configuration',
                        help='Yaml file with the configuration of the benchmark.',
                        type=str,
                        default="~/w/catkin_ws/src/performance_modelling/config/benchmark_configurations/slam_benchmark_1.yaml",
                        required=False)

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder in which the result of each run will be placed.',
                        type=str,
                        default="~/ds/performance_modelling_output/test/",
                        required=False)

    parser.add_argument('-n', '--num-runs', dest='num_runs',
                        help='Number of runs to be executed for each combination of configurations.',
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument('-g', '--headless', dest='headless',
                        help='When set the components are run with no GUI.',
                        action='store_true',
                        required=False)

    parser.add_argument('-s', '--show-ros-info', dest='show_ros_info',
                        help='When set the component nodes are launched with output="screen".',
                        action='store_true',
                        required=False)

    args = parser.parse_args()
    base_run_folder = path.expanduser(args.base_run_folder)
    environment_dataset_folder = path.expanduser(args.environment_dataset_folder)
    benchmark_configuration = path.expanduser(args.benchmark_configuration)

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('performance_modelling')
    components_configurations_folder = path.join(package_path, "config/component_configurations/")

    if not path.exists(base_run_folder):
        os.makedirs(base_run_folder)

    log_file_path = path.join(base_run_folder, "benchmark_log.csv")

    environment_folders = sorted(map(path.dirname, set(glob.glob(path.join(path.abspath(path.expanduser(environment_dataset_folder)), "**/*.world"))).union(set(glob.glob(path.join(path.abspath(path.expanduser(environment_dataset_folder)), "*.world"))))))
    print_info("environments found: {}".format(len(environment_folders)))

    with open(benchmark_configuration, 'r') as f:
        benchmark_configuration = yaml.load(f)

    supervisor_configuration_path = path.join(components_configurations_folder, benchmark_configuration['supervisor_configuration'])

    components_configurations_dict = benchmark_configuration['components_configurations']

    # convert components configurations relative paths to absolute paths
    for component, configurations_relative_path_list in components_configurations_dict.items():
        components_configurations_dict[component] = map(lambda relative_path: path.join(components_configurations_folder, relative_path), configurations_relative_path_list)

    # convert the dict with {component_name: [configuration_1, configuration_2]} to the list [(component_name, configuration_1), (component_name, configuration_2), ...]
    configurations_alternatives = list()
    for component, configurations_list in components_configurations_dict.items():
        component_configuration_list_of_tuples = map(lambda configuration: (component, configuration), configurations_list)
        configurations_alternatives.append(component_configuration_list_of_tuples)

    # obtain the list of combinations from the list of alternatives
    # example: component 1 with configurations [A, B, C, D], and component 2 with configurations [x, y]:
    #   itertools.product([A, B, C, D], [x, y]) --> [[A, x], [A, y], [B, x], [B, y], [C, x], [C, y], [D, x], [D, y]]
    configuration_combinations_lists = list(itertools.product(*configurations_alternatives))

    # convert the list of lists to a list of dicts
    configuration_combinations_dicts = map(dict, configuration_combinations_lists)

    num_combinations = len(configuration_combinations_dicts)
    print_info("number of configuration combinations: {}".format(num_combinations))
    print_info("number of runs per combination: {}".format(args.num_runs))
    print_info("total number of runs: {}".format(args.num_runs * num_combinations))

    for _ in range(args.num_runs):
        for environment_folder in environment_folders:
            for components_configurations in configuration_combinations_dicts:

                # find an available run folder path
                i = 0
                run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))
                while path.exists(run_folder):
                    i += 1
                    run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))

                print_info("benchmark: starting run {run_index}".format(run_index=i))
                print_info("\tenvironment_folder:", environment_folder)
                print_info("\tsupervisor_configuration:", supervisor_configuration_path)
                print_info("\tcomponents_configurations:")
                for k, v in components_configurations.items():
                    print_info("\t\t{}: ...{}".format(k, v[-100:]))

                # instantiate and execute the run
                try:
                    r = BenchmarkRun(run_id=i,
                                     run_output_folder=run_folder,
                                     benchmark_log_path=log_file_path,
                                     show_ros_info=args.show_ros_info,
                                     headless=args.headless,
                                     stage_dataset_folder=environment_folder,
                                     component_configuration_files=components_configurations,
                                     supervisor_configuration_file=supervisor_configuration_path)

                    r.execute_run()

                    if r.ros_has_shutdown:
                        print_info("benchmark: run {run_index} interrupted".format(run_index=i))
                        print_info("benchmark: interrupted")
                        sys.exit(0)
                    else:
                        print_info("benchmark: run {run_index} completed".format(run_index=i))

                except roslaunch.RLException:
                    print_error(traceback.format_exc())
                    sys.exit(0)
                except IOError:
                    print_error(traceback.format_exc())
                except ValueError:
                    print_error(traceback.format_exc())

    print_info("benchmark: finished")
