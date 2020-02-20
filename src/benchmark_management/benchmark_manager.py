#!/usr/bin/python

from __future__ import print_function

import rospy
import sys
from os import path
import os

from component import Component
from run import Run


class BenchmarkManager(object):
    def __init__(self):

        # run parameters
        self.headless = False
        self.map_steady_state_period = 600.0
        self.map_snapshot_period = 30.0
        self.run_timeout = 10800.0

        # run constants
        self.run_output_folder = path.expanduser("~/tmp/run_0/")

        # run variables
        self.map_snapshot_count = 0
        self.map_saver = None
        self.supervisor = None

    def execute_run(self):
        roscore = Component('roscore', 'performance_modelling_ros', 'roscore.launch')
        rviz = Component('rviz', 'performance_modelling_ros', 'rviz.launch')
        environment = Component('stage', 'performance_modelling_ros', 'stage.launch')
        slam = Component('gmapping', 'performance_modelling_ros', 'gmapping.launch')
        navigation = Component('move_base', 'performance_modelling_ros', 'move_base.launch')
        explorer = Component('explorer', 'performance_modelling_ros', 'explorer.launch')
        self.supervisor = Component('supervisor', 'slam_benchmark_supervisor', 'supervisor.launch')

        stage_world_file = path.expanduser("~/ds/performance_modelling/airlab/airlab.world")
        gmapping_configuration_file = path.expanduser("~/w/catkin_ws/src/performance_modelling_ros/config/gmapping_1.yaml")

        # Prepare folder structure
        if not path.exists(self.run_output_folder):
            print("execute_run: Making dir {dir}".format(dir=self.run_output_folder))
            os.mkdir(self.run_output_folder)
        elif not path.isdir(self.run_output_folder):
            print("execute_run: Output folder path {dir} exists but is not a directory!".format(dir=self.run_output_folder))
            sys.exit(-2)

        # Launch roscore and manager's own node
        roscore.launch()
        rospy.init_node('benchmark_manager_node', anonymous=True)
        run_timeout_timer = rospy.Timer(rospy.Duration.from_sec(self.run_timeout), self.run_timeout_callback)

        # Launch components
        if not self.headless:
            rviz.launch()
        environment.launch(stage_world_file=stage_world_file, headless=self.headless)
        slam.launch(configuration=gmapping_configuration_file)
        navigation.launch()
        explorer.launch(log_path=self.run_output_folder)
        self.supervisor.launch(map_steady_state_period=self.map_steady_state_period, map_snapshot_period=self.map_snapshot_period, run_output_folder=self.run_output_folder)

        # Wait for supervisor component to finish.
        # Timeout and other criteria can also terminate the supervisor.
        print("execute_run: Waiting for supervisor to finish")
        self.supervisor.wait_to_finish()

        # Destroy timers and shutdown remaining components
        run_timeout_timer.shutdown()

        if rospy.is_shutdown():
            print("execute_run: ROS has shutdown")
        else:
            if not self.headless:
                rviz.shutdown()
            explorer.shutdown()
            navigation.shutdown()
            slam.shutdown()
            environment.shutdown()
            roscore.shutdown()
            print("execute_run: Shutdown completed")

    def run_timeout_callback(self, _):
        if rospy.is_shutdown():
            print("run_timeout_callback: ROS has shutdown")
            return
        if self.supervisor is None:
            print("run_timeout_callback: self.supervisor was None, weird!")
            return

        print("run_timeout_callback: Shutting down supervisor due to timeout")
        self.supervisor.shutdown()


if __name__ == '__main__':
    BenchmarkManager().execute_run()
