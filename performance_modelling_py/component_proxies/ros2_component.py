# -*- coding: utf-8 -*-
import subprocess
import time


class Component(object):

    def __init__(self, name, package_name, launchfile_name, component_parameters=None):
        self.name = name
        self._package_name = package_name
        self._launchfile_name = launchfile_name
        self._component_configuration = component_parameters if component_parameters is not None else dict()

        self._process = None

    def launch(self):
        launch_args = list()
        for arg_name, arg_value in self._component_configuration.items():
            launch_args.append("{arg_name}:={arg_value}".format(arg_name=arg_name, arg_value=str(arg_value)))

        self._process = subprocess.Popen(["ros2", "launch", self._package_name, self._launchfile_name] + launch_args)
        return

    def shutdown(self):
        if self._process is not None:
            self._process.kill()

    def wait_to_finish(self):
        if self._process is not None:
            self._process.wait()

    @property
    def package_name(self):
        return self._package_name

    @property
    def launchfile_name(self):
        return self._launchfile_name
