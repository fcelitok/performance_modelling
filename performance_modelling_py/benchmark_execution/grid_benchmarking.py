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


def execute_grid_benchmark(benchmark_run_object, grid_benchmark_configuration, environment_folders, base_run_folder, num_runs, headless, show_ros_info):

    if not path.exists(base_run_folder):
        os.makedirs(base_run_folder)

    log_file_path = path.join(base_run_folder, "benchmark_log.csv")

    print_info("environments found: {}".format(len(environment_folders)))

    with open(grid_benchmark_configuration, 'r') as f:
        grid_benchmark_configuration = yaml.load(f)

    combinatorial_parameters_dict = grid_benchmark_configuration['combinatorial_parameters']

    # convert the dict with {parameter_name_1: [parameter_value_1, parameter_value_2], ...}
    # to the list [(parameter_name_1, parameter_value_1), (parameter_name_1, parameter_value_2), ...]
    parameters_alternatives = list()
    for parameter_name, parameter_values_list in combinatorial_parameters_dict.items():
        parameter_list_of_tuples = list(map(lambda parameter_value: (parameter_name, parameter_value), parameter_values_list))
        parameters_alternatives.append(parameter_list_of_tuples)

    # obtain the list of combinations from the list of alternatives
    # ex: itertools.product([(a, 1), (a, 2)], [(b, 0.1), (b, 0.2)]) --> [ [(a, 1), (b, 0.1)], [(a, 1), (b, 0.2)], [(a, 2), (b, 0.1)], [(a, 2), (b, 0.2)] ]
    parameters_combinations_lists = list(itertools.product(*parameters_alternatives))

    # convert the list of lists to a list of dicts
    parameters_combinations_dict_list = list(map(dict, parameters_combinations_lists))

    num_combinations = len(parameters_combinations_dict_list)
    num_environments = len(environment_folders)
    print_info("number of environments:           {}".format(num_environments))
    print_info("number of parameter combinations: {}".format(num_combinations))
    print_info("number of repetition runs:        {}".format(num_runs))
    print_info("total number of runs:             {}".format(num_runs * num_combinations * num_environments))

    for _ in range(num_runs):
        for parameters_combination_dict in parameters_combinations_dict_list:
            for environment_folder in environment_folders:

                # find an available run folder path
                i = 0
                run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))
                while path.exists(run_folder):
                    i += 1
                    run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))

                print_info("\n\n\nbenchmark: starting run {run_index}".format(run_index=i))
                print_info("\tenvironment_folder:", environment_folder)
                print_info("\tparameters_combination_dict:")
                for k, v in parameters_combination_dict.items():
                    print_info("\t\t{}: {}".format(k, v))

                # instantiate and execute the run
                # noinspection PyBroadException
                try:
                    r = benchmark_run_object(
                        run_id=i,
                        run_output_folder=run_folder,
                        benchmark_log_path=log_file_path,
                        environment_folder=environment_folder,
                        parameters_combination_dict=parameters_combination_dict,
                        benchmark_configuration_dict=grid_benchmark_configuration,
                        show_ros_info=show_ros_info,
                        headless=headless,
                    )

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
