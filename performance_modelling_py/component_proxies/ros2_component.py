# -*- coding: utf-8 -*-
import asyncio
from os import path
from sys import argv
from typing import Optional

import launch
import osrf_pycommon
import osrf_pycommon.process_utils
from launch import LaunchService, LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from performance_modelling_py.utils import print_error


class Component(object):

    _launch_service: Optional[launch.LaunchService]

    def __init__(self, name, package_name, launchfile_name, component_parameters=None):
        self.name = name
        self._package_name = package_name
        self._launchfile_name = launchfile_name

        self._launch_arguments = list()
        if component_parameters is not None:
            # IncludeLaunchDescription requires the arguments as a list and the values as strings, so the values are converted to strings
            for arg_key, arg_value in component_parameters.items():
                self._launch_arguments.append((arg_key, str(arg_value)))

        launchfile_path = path.join(get_package_share_directory(self._package_name), "launch", self._launchfile_name)
        launch_description_source = PythonLaunchDescriptionSource(launchfile_path)

        ld = LaunchDescription([
            IncludeLaunchDescription(launch_description_source, launch_arguments=self._launch_arguments)
        ])

        self._launch_service = LaunchService(argv=argv)
        self._launch_service.include_launch_description(ld)
        self._loop = osrf_pycommon.process_utils.get_loop()

        self._launch_task = None

    def launch(self):
        if self._launch_task is not None:
            print_error("Component.launch: component {} was already launched".format(self.name))

        self._launch_task = self._loop.create_task(self._launch_service.run_async())

    def launch_and_wait_to_finish(self):
        if self._launch_task is not None:
            print_error("Component.launch_and_wait_to_finish: component {} was already launched".format(self.name))

        self._launch_task = self._loop.create_task(self._launch_service.run_async())
        self._loop.run_until_complete(self._launch_task)

    def shutdown(self):
        if self._launch_task is None:
            print_error("Component.shutdown: component {} has not been launched".format(self.name))
            return

        if not self._launch_task.done():
            asyncio.ensure_future(self._launch_service.shutdown(), loop=self._loop)
            self._loop.run_until_complete(self._launch_task)

    @property
    def package_name(self):
        return self._package_name

    @property
    def launchfile_name(self):
        return self._launchfile_name
