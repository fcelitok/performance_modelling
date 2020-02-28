#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import rospkg
import argparse
import os
import yaml
from os import path
import itertools

from performance_modelling_ros import Component
from performance_modelling_ros.utils import backup_file_if_exists, print_info
from compute_metrics import compute_localization_metrics
from visualisation import save_trajectories_plot


class BenchmarkRun(object):
    def __init__(self, run_output_folder, show_ros_info, headless, stage_world_file, component_configuration_files, supervisor_configuration_file):

        # components configuration parameters
        self.move_base_configuration_file = component_configuration_files['move_base']
        self.gmapping_configuration_file = component_configuration_files['gmapping']
        self.explore_lite_configuration_file = component_configuration_files['explore_lite']
        self.supervisor_configuration_file = supervisor_configuration_file

        # environment parameters
        self.stage_world_file = stage_world_file

        # run parameters
        self.headless = headless
        self.components_ros_output = 'screen' if show_ros_info else 'log'
        self.run_output_folder = run_output_folder

    def execute_run(self):
        # prepare folder structure
        backup_file_if_exists(self.run_output_folder)
        os.mkdir(self.run_output_folder)
        bag_file_path = path.join(self.run_output_folder, "odom_tf_ground_truth.bag")

        # components parameters
        Component.common_parameters = {'headless': self.headless, 'output': self.components_ros_output}
        environment_params = {'stage_world_file': self.stage_world_file}
        recorder_params = {'bag_file_path': bag_file_path}
        slam_params = {'configuration': self.gmapping_configuration_file}
        explorer_params = {'configuration': self.explore_lite_configuration_file}
        navigation_params = {'configuration': self.move_base_configuration_file}
        supervisor_params = {'run_output_folder': self.run_output_folder, 'configuration': self.supervisor_configuration_file}

        # declare components
        roscore = Component('roscore', 'performance_modelling', 'roscore.launch')
        rviz = Component('rviz', 'performance_modelling', 'rviz.launch')
        environment = Component('stage', 'performance_modelling', 'stage.launch', environment_params)
        recorder = Component('recorder', 'performance_modelling', 'rosbag_recorder.launch', recorder_params)
        slam = Component('gmapping', 'performance_modelling', 'gmapping.launch', slam_params)
        navigation = Component('move_base', 'performance_modelling', 'move_base.launch', navigation_params)
        explorer = Component('explore_lite', 'performance_modelling', 'explore_lite.launch', explorer_params)
        supervisor = Component('supervisor', 'performance_modelling', 'slam_benchmark_supervisor.launch', supervisor_params)

        # launch components
        print_info("execute_run: launching components")
        roscore.launch()
        rviz.launch()
        environment.launch()
        recorder.launch()
        slam.launch()
        navigation.launch()
        explorer.launch()
        supervisor.launch()

        # TODO check if all components launched properly

        # wait for the supervisor component to finish
        print_info("execute_run: waiting for supervisor to finish")
        supervisor.wait_to_finish()
        print_info("execute_run: supervisor has shutdown")

        # shutdown remaining components
        explorer.shutdown()
        navigation.shutdown()
        slam.shutdown()
        recorder.shutdown()
        environment.shutdown()
        rviz.shutdown()
        roscore.shutdown()
        print_info("execute_run: components shutdown completed")

        compute_localization_metrics(self.run_output_folder)
        print_info("execute_run: metrics computation completed")

        save_trajectories_plot(self.run_output_folder)
        print_info("execute_run: saved visualisation files")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the benchmark')

    parser.add_argument('-e', dest='environment_dataset_folder',
                        help='Dataset folder containg the stage environment.world file.',
                        type=str,
                        default="~/ds/performance_modelling/airlab/",
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

    with open(benchmark_configuration, 'r') as f:
        benchmark_configuration = yaml.load(f)

    supervisor_configuration = path.join(components_configurations_folder, benchmark_configuration['supervisor_configuration'])

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

    for components_configurations in configuration_combinations_dicts:
        for _ in range(args.num_runs):

            # find an available run folder path
            i = 0
            run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))
            while path.exists(run_folder):
                i += 1
                run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))

            # instantiate and execute the run
            print_info("benchmark: starting run {run_index}".format(run_index=i))
            r = BenchmarkRun(run_output_folder=run_folder,
                             show_ros_info=args.show_ros_info,
                             headless=args.headless,
                             stage_world_file=path.join(environment_dataset_folder, "environment.world"),
                             component_configuration_files=components_configurations,
                             supervisor_configuration_file=supervisor_configuration)

            r.execute_run()
            print_info("benchmark: run {run_index} completed".format(run_index=i))

    print_info("benchmark: all runs completed")
