#!/usr/bin/python

from __future__ import print_function

from component import *
from run import *


class BenchmarkManager(object):
    def __init__(self):
        pass

    @staticmethod
    def test():
        c = Component('test_gmapping', 'test', 'gmapping.launch')

        print(c.launchfile_path)

        c.launch(launch_args=['configuration:=/home/enrico/w/catkin_ws/src/test/config/gmapping_1.yaml', 'test_arg:=ASDASDASD'])


BenchmarkManager().test()
