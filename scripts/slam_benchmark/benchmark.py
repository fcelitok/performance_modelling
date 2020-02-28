#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os
from os import path

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

        # run variables
        self.supervisor = None

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
        self.supervisor = Component('supervisor', 'slam_benchmark_supervisor', 'supervisor.launch', supervisor_params)

        # launch components
        print_info("execute_run: launching components")
        roscore.launch()
        rviz.launch()
        environment.launch()
        recorder.launch()
        slam.launch()
        navigation.launch()
        explorer.launch()
        self.supervisor.launch()

        # TODO check if all components launched properly

        # wait for the supervisor component to finish
        print_info("execute_run: waiting for supervisor to finish")
        self.supervisor.wait_to_finish()
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
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Runs slam benchmark')

    parser.add_argument('-e', dest='environment_dataset_folder',
                        help='Dataset folder containg the stage environment.world file.',
                        type=str,
                        default="~/ds/performance_modelling/airlab/",
                        required=False)

    parser.add_argument('-c', dest='components_configuration_folder',
                        help='Folder containing the configuration for each component.',
                        type=str,
                        default="~/w/catkin_ws/src/performance_modelling/config/components/",
                        required=False)

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder in which the result of each run will be placed.',
                        type=str,
                        default="~/ds/performance_modelling_output/test/",
                        required=False)

    parser.add_argument('-n', '--num-runs', dest='num_runs',
                        help='Number of runs to be executed.',
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
    components_configuration_folder = path.expanduser(args.components_configuration_folder)

    if not path.exists(base_run_folder):
        os.makedirs(base_run_folder)

    for _ in range(args.num_runs):

        i = 0
        run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))
        while path.exists(run_folder):
            i += 1
            run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))

        component_configurations = {
            'gmapping': path.join(components_configuration_folder, "gmapping/gmapping_1.yaml"),
            'move_base': path.join(components_configuration_folder, "move_base/move_base_1.yaml"),
            'explore_lite': path.join(components_configuration_folder, "explore_lite/explore_lite_1.yaml"),
        }

        supervisor_configuration = path.join(components_configuration_folder, "slam_benchmark_supervisor/slam_benchmark_supervisor.yaml")

        r = BenchmarkRun(run_output_folder=run_folder,
                         show_ros_info=args.show_ros_info,
                         headless=args.headless,
                         stage_world_file=path.join(environment_dataset_folder, "environment.world"),
                         component_configuration_files=component_configurations,
                         supervisor_configuration_file=supervisor_configuration)

        r.execute_run()
        print_info("benchmark: run {run_index} completed".format(run_index=i))

    print_info("benchmark: all runs completed")
