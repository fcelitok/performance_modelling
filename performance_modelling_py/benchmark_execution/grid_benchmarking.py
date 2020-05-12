#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import traceback
import yaml
from os import path
import itertools

from performance_modelling_py.utils import print_info, print_fatal


def execute_grid_benchmark(benchmark_run_object, grid_benchmark_configuration, components_configurations_folder, environment_folders, base_run_folder, num_runs, headless, show_ros_info):

    if not path.exists(base_run_folder):
        os.makedirs(base_run_folder)

    log_file_path = path.join(base_run_folder, "benchmark_log.csv")

    print_info("environments found: {}".format(len(environment_folders)))

    with open(grid_benchmark_configuration, 'r') as f:
        grid_benchmark_configuration = yaml.load(f)

    supervisor_configuration_path = path.join(components_configurations_folder, grid_benchmark_configuration['supervisor_configuration'])

    components_configurations_dict = grid_benchmark_configuration['components_configurations']

    # convert components configurations relative paths to absolute paths
    for component, configurations_relative_path_list in components_configurations_dict.items():
        components_configurations_dict[component] = list(map(lambda relative_path: path.join(components_configurations_folder, relative_path), configurations_relative_path_list))

    # convert the dict with {component_name: [configuration_1, configuration_2]} to the list [(component_name, configuration_1), (component_name, configuration_2), ...]
    configurations_alternatives = list()
    for component, configurations_list in components_configurations_dict.items():
        component_configuration_list_of_tuples = list(map(lambda configuration: (component, configuration), configurations_list))
        configurations_alternatives.append(component_configuration_list_of_tuples)

    # obtain the list of combinations from the list of alternatives
    # example: component_1 with configurations [A, B, C, D], and component_2 with configurations [x, y]:
    #   itertools.product([A, B, C, D], [x, y]) --> [[A, x], [A, y], [B, x], [B, y], [C, x], [C, y], [D, x], [D, y]]
    #   itertools.product([(gmapping, gmapping_1.yaml), (gmapping, gmapping_2.yaml)], [(move_base, move_base_1.yaml), (move_base, move_base_2.yaml)])
    #   --> [ [(gmapping, gmapping_1.yaml), (move_base, move_base_1.yaml)], [(gmapping, gmapping_1.yaml), (move_base, move_base_2.yaml)], [(gmapping, gmapping_2.yaml), (move_base, move_base_1.yaml)], [(gmapping, gmapping_2.yaml), (move_base, move_base_2.yaml)] ]
    configuration_combinations_lists = list(itertools.product(*configurations_alternatives))

    # convert the list of lists to a list of dicts
    configuration_combinations_dicts = list(map(dict, configuration_combinations_lists))

    num_combinations = len(configuration_combinations_dicts)
    num_environments = len(environment_folders)
    print_info("number of environments:               {}".format(num_environments))
    print_info("number of configuration combinations: {}".format(num_combinations))
    print_info("number of repetition runs:            {}".format(num_runs))
    print_info("total number of runs:                 {}".format(num_runs * num_combinations * num_environments))

    for _ in range(num_runs):
        for components_configurations in configuration_combinations_dicts:
            for environment_folder in environment_folders:

                # find an available run folder path
                i = 0
                run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))
                while path.exists(run_folder):
                    i += 1
                    run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))

                print_info("\n\n\nbenchmark: starting run {run_index}".format(run_index=i))
                print_info("\tenvironment_folder:", environment_folder)
                print_info("\tsupervisor_configuration:", supervisor_configuration_path)
                print_info("\tcomponents_configurations:")
                for k, v in components_configurations.items():
                    print_info("\t\t{}: ...{}".format(k, v[-100:]))

                # instantiate and execute the run
                # noinspection PyBroadException
                try:
                    r = benchmark_run_object(run_id=i,
                                             run_output_folder=run_folder,
                                             benchmark_log_path=log_file_path,
                                             show_ros_info=show_ros_info,
                                             headless=headless,
                                             environment_folder=environment_folder,
                                             component_configuration_file_paths=components_configurations,
                                             supervisor_configuration_file_path=supervisor_configuration_path)

                    r.execute_run()

                    if r.aborted:
                        print_info("benchmark: run {run_index} aborted".format(run_index=i))
                        print_info("benchmark: interrupted")
                        sys.exit(0)
                    else:
                        print_info("benchmark: run {run_index} completed".format(run_index=i))

                except IOError:
                    print_fatal(traceback.format_exc())
                except ValueError:
                    print_fatal(traceback.format_exc())
                except ZeroDivisionError:
                    print_fatal(traceback.format_exc())
                except:
                    print_fatal(traceback.format_exc())
                    sys.exit(0)

    print_info("benchmark: finished")
