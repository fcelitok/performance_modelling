import rospy
import roslaunch
import roslaunch.rlutil
import roslaunch.parent


class Component(object):

    def __init__(self, name, package_name, launchfile_name):
        self.name = name
        self.package_name = package_name
        self.launchfile_name = launchfile_name
        self.launchfile_path = roslaunch.rlutil.resolve_launch_arguments([self.package_name, self.launchfile_name])[0]

    def launch(self, configuration=None, launch_args=None):
        assert(configuration is None or isinstance(configuration, Configuration))
        assert(launch_args is None or isinstance(launch_args, list))

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [(self.launchfile_path, launch_args)])
        launch.start()
        rospy.loginfo("Component {} started")


class Configuration(object):
    def __init__(self):
        pass
