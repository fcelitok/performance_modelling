#!/usr/bin/python

from __future__ import print_function

import roslib
import roslib.packages
import sys
import os
from os import path

from performance_modelling_ros import Component
from compute_metrics import generate_all


def backup_file_if_exists(target_path):
    if path.exists(target_path):
        backup_path = path.abspath(target_path) + '.backup'
        backup_file_if_exists(backup_path)
        print("backing up: {} -> {}".format(target_path, backup_path))
        os.rename(target_path, backup_path)


class BenchmarkRun(object):
    def __init__(self, run_output_folder, stage_world_file, gmapping_configuration_file, headless):

        # run parameters
        self.run_output_folder = run_output_folder
        self.gmapping_configuration_file = gmapping_configuration_file
        self.stage_world_file = stage_world_file
        self.headless = headless
        self.map_steady_state_period = 60.0
        self.map_snapshot_period = 5.0
        self.run_timeout = 10800.0

        # Find the metric_evaluator executable
        metric_evaluator_package_name = 'performance_modelling'
        metric_evaluator_exec_name = 'metricEvaluator'
        metric_evaluator_resources_list = roslib.packages.find_resource(metric_evaluator_package_name, metric_evaluator_exec_name)
        if len(metric_evaluator_resources_list) > 1:
            print("Multiple files named [{resource_name}}] in package [{package_name}]:%s".format(resource_name=metric_evaluator_exec_name, package_name=metric_evaluator_package_name))
        elif len(metric_evaluator_resources_list) == 0:
            print("No files named [{resource_name}}] in package [{package_name}]:%s".format(resource_name=metric_evaluator_exec_name, package_name=metric_evaluator_package_name))
        else:
            self.metric_evaluator_exec_path = metric_evaluator_resources_list[0]

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
        explorer = Component('explorer', components_pkg, 'explorer.launch')
        self.supervisor = Component('supervisor', 'slam_benchmark_supervisor', 'supervisor.launch')

        # Prepare folder structure
        backup_file_if_exists(self.run_output_folder)
        os.mkdir(self.run_output_folder)
        bag_file_path = path.join(self.run_output_folder, "odom_tf_ground_truth.bag")

        # Launch roscore
        roscore.launch()

        # Launch components
        rviz.launch(headless=self.headless)
        environment.launch(stage_world_file=self.stage_world_file,
                           headless=self.headless)
        recorder.launch(bag_file_path=bag_file_path)
        slam.launch(configuration=self.gmapping_configuration_file)
        navigation.launch()
        explorer.launch(log_path=self.run_output_folder)
        self.supervisor.launch(run_timeout=self.run_timeout,
                               write_base_link_poses_period=0.01,
                               map_steady_state_period=self.map_steady_state_period,
                               map_snapshot_period=self.map_snapshot_period,
                               run_output_folder=self.run_output_folder,
                               map_change_threshold=10.0,
                               size_change_threshold=5.0)

        # Wait for supervisor component to finish.
        # Timeout and other criteria can also terminate the supervisor.
        print("execute_run: Waiting for supervisor to finish")
        self.supervisor.wait_to_finish()
        print("execute_run: supervisor has shutdown")

        # Shutdown remaining components
        rviz.shutdown()
        explorer.shutdown()
        navigation.shutdown()
        slam.shutdown()
        recorder.shutdown()
        environment.shutdown()
        roscore.shutdown()
        print("execute_run: Shutdown completed")

        generate_all(self.run_output_folder, self.metric_evaluator_exec_path)


if __name__ == '__main__':
    for i in range(10):
        r = BenchmarkRun(run_output_folder=path.expanduser("~/tmp/run_{run_number}/".format(run_number=i)),
                         stage_world_file=path.expanduser("~/ds/performance_modelling/airlab/airlab.world"),
                         gmapping_configuration_file=path.expanduser("~/w/catkin_ws/src/performance_modelling/config/components/gmapping_1.yaml"),
                         headless=False)

        r.execute_run()
