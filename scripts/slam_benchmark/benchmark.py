#!/usr/bin/python

from __future__ import print_function

import argparse
import os
from os import path

from performance_modelling_ros import Component
from compute_metrics import compute_localization_metrics
from visualisation import save_trajectories_plot


def backup_file_if_exists(target_path):
    if path.exists(target_path):
        backup_path = path.abspath(target_path) + '.backup'
        backup_file_if_exists(backup_path)
        print("backing up: {} -> {}".format(target_path, backup_path))
        os.rename(target_path, backup_path)


class BenchmarkRun(object):
    def __init__(self, run_output_folder, stage_world_file, gmapping_configuration_file, headless,
                 local_planner_configuration_file, global_planner_configuration_file, show_ros_info):

        # component configuration parameters
        self.local_planner_configuration_file = local_planner_configuration_file
        self.global_planner_configuration_file = global_planner_configuration_file
        self.gmapping_configuration_file = gmapping_configuration_file
        self.components_ros_output = 'screen' if show_ros_info else 'log'

        # environment parameters
        self.stage_world_file = stage_world_file

        # run parameters
        self.run_output_folder = run_output_folder
        self.headless = headless
        self.map_steady_state_period = 60.0
        self.map_snapshot_period = 5.0
        self.run_timeout = 10800.0

        # run variables
        self.map_snapshot_count = 0
        self.map_saver = None
        self.supervisor = None

    def execute_run(self):
        components_pkg = 'performance_modelling'
        roscore = Component('roscore', components_pkg, 'roscore.launch')
        rviz = Component('rviz', components_pkg, 'rviz.launch')
        environment = Component('stage', components_pkg, 'stage.launch')
        recorder = Component('recorder', components_pkg, 'rosbag_recorder.launch')
        slam = Component('gmapping', components_pkg, 'gmapping.launch')
        navigation = Component('move_base', components_pkg, 'move_base.launch')
        explorer = Component('explorer', components_pkg, 'lite_explorer.launch')
        self.supervisor = Component('supervisor', 'slam_benchmark_supervisor', 'supervisor.launch')

        # prepare folder structure
        backup_file_if_exists(self.run_output_folder)
        os.mkdir(self.run_output_folder)
        bag_file_path = path.join(self.run_output_folder, "odom_tf_ground_truth.bag")

        # launch components
        print("execute_run: launching components")
        roscore.launch()
        rviz.launch(headless=self.headless,
                    output=self.components_ros_output)
        environment.launch(stage_world_file=self.stage_world_file,
                           headless=self.headless,
                           output=self.components_ros_output)
        recorder.launch(bag_file_path=bag_file_path,
                        output=self.components_ros_output)
        slam.launch(configuration=self.gmapping_configuration_file,
                    output=self.components_ros_output)
        navigation.launch(local_planner_configuration=self.local_planner_configuration_file,
                          global_planner_configuration=self.global_planner_configuration_file,
                          output=self.components_ros_output)
        explorer.launch(output=self.components_ros_output)
        self.supervisor.launch(run_output_folder=self.run_output_folder,
                               run_timeout=self.run_timeout,
                               map_steady_state_period=self.map_steady_state_period,
                               map_snapshot_period=self.map_snapshot_period,
                               map_change_threshold=10.0,
                               map_size_change_threshold=5.0,
                               map_occupied_threshold=65,
                               map_free_threshold=25,
                               write_base_link_poses_period=0.1)

        # TODO check if all components launched properly

        # wait for the supervisor component to finish.
        print("execute_run: waiting for supervisor to finish")
        self.supervisor.wait_to_finish()
        print("execute_run: supervisor has shutdown")

        # shutdown remaining components
        rviz.shutdown()
        explorer.shutdown()
        navigation.shutdown()
        slam.shutdown()
        recorder.shutdown()
        environment.shutdown()
        roscore.shutdown()
        print("execute_run: components shutdown completed")

        compute_localization_metrics(self.run_output_folder)
        print("execute_run: metrics computation completed")

        save_trajectories_plot(self.run_output_folder)
        print("execute_run: saved visualisation files")


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
                        type=bool,
                        default=False,
                        required=False)

    parser.add_argument('-s', '--show-ros-info', dest='show_ros_info',
                        help='When set the component nodes are launched with output="screen".',
                        type=bool,
                        default=False,
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

        r = BenchmarkRun(run_output_folder=run_folder,
                         stage_world_file=path.join(environment_dataset_folder, "environment.world"),
                         gmapping_configuration_file=path.join(components_configuration_folder, "gmapping/gmapping_1.yaml"),
                         local_planner_configuration_file=path.join(components_configuration_folder, "move_base/local_planner_1.yaml"),
                         global_planner_configuration_file=path.join(components_configuration_folder, "move_base/global_planner_1.yaml"),
                         headless=args.headless,
                         show_ros_info=args.show_ros_info)

        r.execute_run()

    print("benchmark: all runs completed")
