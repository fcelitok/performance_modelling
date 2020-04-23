# -*- coding: utf-8 -*-

import roslaunch
import roslaunch.rlutil
import roslaunch.parent


class Component(object):

    common_parameters = dict()

    def __init__(self, name, package_name, launchfile_name, component_parameters=None):
        self.name = name
        self._package_name = package_name
        self._launchfile_name = launchfile_name
        self._launchfile_path = None  # TODO
        self._component_configuration = component_parameters if component_parameters is not None else dict()

        # TODO

    def launch(self, **additional_parameters):

        this_component_parameters = dict()

        for arg_name, arg_value in Component.common_parameters.items():
            this_component_parameters[arg_name] = arg_value

        # component parameters overwrite the common parameters
        for arg_name, arg_value in self._component_configuration.items():
            this_component_parameters[arg_name] = arg_value

        # additional parameters overwrite the common and component parameters
        for arg_name, arg_value in additional_parameters.items():
            this_component_parameters[arg_name] = arg_value

        # TODO

    def wait_to_finish(self):
        pass  # TODO

    def shutdown(self):
        pass  # TODO

    @property
    def package_name(self):
        return self._package_name

    @property
    def launchfile_name(self):
        return self._launchfile_name

    @property
    def launchfile_path(self):
        return self._launchfile_path
