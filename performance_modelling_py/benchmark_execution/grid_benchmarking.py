#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
import random
import sys
import traceback
from collections import defaultdict
from datetime import datetime

import yaml
from os import path
import itertools

from performance_modelling_py.utils import print_info, print_fatal, print_error


class hashable_dict(dict):
    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


def execute_grid_benchmark(benchmark_run_object, grid_benchmark_configuration, environment_folders, base_run_folder, num_runs, ignore_executed_params_combinations, shuffle, headless, show_ros_info):

    if not path.exists(base_run_folder):
        os.makedirs(base_run_folder)
        # base_run_folder: ~/ds/performance_modelling/output/test_planning/

    log_file_path = path.join(base_run_folder, "benchmark_log.csv")

    with open(grid_benchmark_configuration, 'r') as f:
        grid_benchmark_configuration = yaml.safe_load(f)
        # grid_benchmark_configuration: ~/turtlebot3_melodic_ws/src/global_planning_performance_modelling/config/benchmark_configurations/global_planning_grid_benchmark_1.yaml"

    combinatorial_parameters_dict = grid_benchmark_configuration['combinatorial_parameters']

    # add environment_name to the parameters and populate the environment_folders_by_name lookup dict
    environment_folders_by_name = dict()
    combinatorial_parameters_dict['environment_name'] = list()
    for environment_folder in environment_folders:
        environment_name = path.basename(path.abspath(environment_folder))
        environment_folders_by_name[environment_name] = environment_folder
        combinatorial_parameters_dict['environment_name'].append(environment_name)

    # convert the dict with {parameter_name_1: [parameter_value_1, parameter_value_2], ...}
    # to the list [(parameter_name_1, parameter_value_1), (parameter_name_1, parameter_value_2), ...] 
    # just separating parameters and its value
    parameters_alternatives = list()
    for parameter_name, parameter_values_list in combinatorial_parameters_dict.items():
        parameter_list_of_tuples = list(map(lambda parameter_value: (parameter_name, parameter_value), parameter_values_list))
        parameters_alternatives.append(parameter_list_of_tuples)

    # obtain the list of combinations from the list of alternatives
    # ex: itertools.product([(a, 1), (a, 2)], [(b, 0.1), (b, 0.2)]) --> [ [(a, 1), (b, 0.1)], [(a, 1), (b, 0.2)], [(a, 2), (b, 0.1)], [(a, 2), (b, 0.2)] ]
    parameters_combinations_lists = list(itertools.product(*parameters_alternatives))

    # convert the list of lists to a list of dicts
    parameters_combinations_dict_list = list(map(dict, parameters_combinations_lists))

    executed_params_combinations = defaultdict(int)
    run_folders = sorted(list(filter(path.isdir, glob.glob(path.abspath(base_run_folder) + '/*'))))
    if not ignore_executed_params_combinations:
        for i, run_folder in enumerate(run_folders):
            run_info_file_path = path.join(run_folder, "run_info.yaml")
            if path.exists(run_info_file_path):
                # noinspection PyBroadException
                try:
                    with open(run_info_file_path) as run_info_file:
                        run_info = yaml.safe_load(run_info_file)
                        params_dict = run_info['run_parameters']
                        params_dict['environment_name'] = path.basename(path.abspath(run_info['environment_folder']))
                        for param_name, param_value in params_dict.items():
                            params_dict[param_name] = tuple(param_value) if type(param_value) == list else param_value  # convert any list into a tuple to allow hashing
                        params_hashable_dict = hashable_dict(params_dict)
                        executed_params_combinations[params_hashable_dict] += 1
                except:
                    print_error(traceback.format_exc())

    remaining_params_combinations = list()
    for parameters_combination_dict in parameters_combinations_dict_list:
        parameters_combination_dict_copy = parameters_combination_dict.copy()
        for param_name, param_value in parameters_combination_dict_copy.items():
            parameters_combination_dict_copy[param_name] = tuple(param_value) if type(param_value) == list else param_value  # convert any list into a tuple to allow hashing
        parameters_combination_hashable_dict = hashable_dict(parameters_combination_dict_copy)
        executed_repetitions = executed_params_combinations[parameters_combination_hashable_dict]
        num_remaining_runs = num_runs - executed_repetitions
        if num_remaining_runs > 0:
            remaining_params_combinations += [parameters_combination_dict] * num_remaining_runs

    num_combinations = len(parameters_combinations_dict_list)
    num_runs_remaining = len(remaining_params_combinations)
    num_executed_runs = len(run_folders)
    num_executed_combinations = len(filter(lambda x: x > 0, executed_params_combinations.values()))

    if ignore_executed_params_combinations:
        print_info("ignoring previous runs")
    else:
        print_info("found {num_executed_runs} executed runs with {num_executed_combinations} params combinations".format(num_executed_runs=num_executed_runs, num_executed_combinations=num_executed_combinations))
    print_info("number of parameter combinations: {}".format(num_combinations))
    print_info("number of repetition runs:        {}".format(num_runs))
    print_info("total number of runs:             {}".format(num_runs * num_combinations))
    print_info("remaining number of runs:         {}".format(num_runs_remaining))

    if shuffle:
        # shuffle the remaining params combinations, to avoid executing consecutively the run repetitions with the same combination
        print_info("shuffling remaining params combinations")
        random.shuffle(remaining_params_combinations)
    else:
        print_info("not shuffling remaining params combinations")

    # generate a session id
    session_id = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S_%f')

    for n_executed_runs, parameters_combination_dict in enumerate(remaining_params_combinations):
        environment_folder = environment_folders_by_name[parameters_combination_dict['environment_name']]

        # find an available run folder path
        i = 0
        run_folder = path.join(base_run_folder, "session_{session_id}_run_{run_number:09d}".format(session_id=session_id, run_number=i))
        while path.exists(run_folder):
            i += 1
            run_folder = path.join(base_run_folder, "session_{session_id}_run_{run_number:09d}".format(session_id=session_id, run_number=i))

        print_info("\n\n\nbenchmark: starting run {run_index} ({remaining_runs} remaining)".format(run_index=i, remaining_runs=len(remaining_params_combinations) - n_executed_runs))
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
        except SystemExit as system_exit_ex:
            print_info("System exit code: ", system_exit_ex.code)
            sys.exit(0)
        except:
            print_fatal(traceback.format_exc())
            sys.exit(0)

    print_info("benchmark: finished")
