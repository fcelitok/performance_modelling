#!/usr/bin/python

from __future__ import print_function

import sys
from os import path
import os
import rospy

from component import *
from run import *


class BenchmarkManager(object):
    def __init__(self):
        self.headless = True
        self.map_save_period = 30.0
        self.run_timeout = 10800.0
        self.output_folder = path.expanduser("~/tmp/run_0/")
        self.map_count = 0
        self.map_saver = None
        self.explorer = None

    def execute_run(self):
        roscore = Component('roscore', 'performance_modelling_ros', 'roscore.launch')
        rviz = Component('rviz', 'performance_modelling_ros', 'rviz.launch')
        environment = Component('stage', 'performance_modelling_ros', 'stage.launch')
        slam = Component('gmapping', 'performance_modelling_ros', 'gmapping.launch')
        navigation = Component('move_base', 'performance_modelling_ros', 'move_base.launch')
        self.explorer = Component('explorer', 'performance_modelling_ros', 'explorer.launch')
        self.map_saver = Component('map_saver', 'performance_modelling_ros', 'map_saver.launch')

        stage_world_file = path.expanduser("~/ds/performance_modelling/airlab/airlab.world")
        gmapping_configuration_file = path.expanduser("~/w/catkin_ws/src/performance_modelling_ros/config/gmapping_1.yaml")

        if not path.exists(self.output_folder):
            print("execute_run: Making dir {dir}".format(dir=self.output_folder))
            os.mkdir(self.output_folder)
        elif not path.isdir(self.output_folder):
            print("execute_run: Output folder path {dir} exists but is not a directory!".format(dir=self.output_folder))
            sys.exit(-2)

        roscore.launch()

        rospy.init_node('benchmark_manager_node', anonymous=True, disable_signals=True)
        map_save_timer = rospy.Timer(rospy.Duration.from_sec(self.map_save_period), self.save_map_callback)
        timeout_timer = rospy.Timer(rospy.Duration.from_sec(self.run_timeout), self.run_timeout_callback)

        if not self.headless:
            rviz.launch()

        environment.launch(stage_world_file=stage_world_file, headless=self.headless)
        slam.launch(configuration=gmapping_configuration_file)
        navigation.launch()
        self.explorer.launch(log_path=self.output_folder)

        print("execute_run: Waiting for explorer to finish")
        self.explorer.wait_to_finish()

        if rospy.is_shutdown():
            print("execute_run: ROS has shutdown")
        else:
            print("execute_run: Saving one last map")
            map_save_timer.shutdown()
            self.save_map_callback()
            print("execute_run: One last map saved")

            rviz.shutdown()
            navigation.shutdown()
            slam.shutdown()
            environment.shutdown()
            roscore.shutdown()

            print("execute_run: Shutdown completed")

    def run_timeout_callback(self, _=None):
        if self.explorer is None:
            print("run_timeout_callback: self.explorer was None, weird!")
            sys.exit(-1)

        print("run_timeout_callback: Shutting down explorer due to timeout")
        self.explorer.shutdown()

    def save_map_callback(self, _=None):
        map_output_file = path.join(self.output_folder, "map_{i}".format(i=self.map_count))
        self.map_count += 1

        self.map_saver.launch(base_filename=map_output_file)


if __name__ == '__main__':
    BenchmarkManager().execute_run()
