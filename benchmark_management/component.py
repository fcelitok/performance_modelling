import rospy
import roslaunch
import roslaunch.rlutil
import roslaunch.parent


class Component(object):

    def __init__(self, name, package_name, launchfile_name):
        self.name = name
        self._package_name = package_name
        self._launchfile_name = launchfile_name
        self._launchfile_path = roslaunch.rlutil.resolve_launch_arguments([self._package_name, self._launchfile_name])[0]

        self.launch_parent = None

    def launch(self, **args):
        launch_args = list()

        for arg_value in args.items():
            launch_args.append(':='.join(arg_value))

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launch_parent = roslaunch.parent.ROSLaunchParent(uuid, [(self._launchfile_path, launch_args)])
        self.launch_parent.start()

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
