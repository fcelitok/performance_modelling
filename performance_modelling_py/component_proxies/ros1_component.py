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
        self._launchfile_path = roslaunch.rlutil.resolve_launch_arguments([self._package_name, self._launchfile_name])[0]
        self._component_configuration = component_parameters if component_parameters is not None else dict()

        self._uuid = None
        self.launch_parent = None

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

        launch_args = list()
        for arg_name, arg_value in this_component_parameters.items():
            launch_args.append('{arg_name}:={arg_value}'.format(arg_name=arg_name, arg_value=str(arg_value)))

        if self._uuid is None:
            self._uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)

        roslaunch.configure_logging(self._uuid)
        self.launch_parent = roslaunch.parent.ROSLaunchParent(self._uuid, [(self._launchfile_path, launch_args)])
        self.launch_parent.start()

    def wait_to_finish(self):
        self.launch_parent.spin()

    def shutdown(self):
        self.launch_parent.shutdown()

    @property
    def package_name(self):
        return self._package_name

    @property
    def launchfile_name(self):
        return self._launchfile_name

    @property
    def launchfile_path(self):
        return self._launchfile_path
