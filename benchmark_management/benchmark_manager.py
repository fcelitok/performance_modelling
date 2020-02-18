#!/usr/bin/python

from __future__ import print_function

from component import *
from run import *


class BenchmarkManager(object):
    def __init__(self):
        pass


def test():
    roscore = Component('roscore', 'test', 'roscore.launch')
    environment = Component('environment_stage', 'test', 'stage.launch')
    slam = Component('slam_gmapping', 'test', 'gmapping.launch')
    navigation = Component('navigation_move_base', 'test', 'move_base.launch')

    print("Components:")
    print(' -', roscore.name, roscore.launchfile_path)
    print(' -', environment.name, environment.launchfile_path)
    print(' -', slam.name, slam.launchfile_path)
    print(' -', navigation.name, navigation.launchfile_path)

    roscore.launch()
    environment.launch()
    slam.launch(configuration='/home/enrico/w/catkin_ws/src/test/config/gmapping_1.yaml', test_arg='test_value')
    navigation.launch()

    print("Launch completed")

    navigation.shutdown()
    slam.shutdown()
    environment.shutdown()
    roscore.shutdown()

    print("Shutdown completed")


if __name__ == '___main___':
    test()
