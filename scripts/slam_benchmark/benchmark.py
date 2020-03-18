#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import rospkg
import argparse
import os
import shutil
import sys
import traceback
import yaml
import time
from os import path
import itertools

import roslaunch
import rospy

from performance_modelling_ros.utils import backup_file_if_exists, print_info, print_error
from performance_modelling_ros import Component
from performance_modelling_ros.metrics.localization_metrics import compute_localization_metrics
from performance_modelling_ros.metrics.map_metrics import compute_map_metrics
from performance_modelling_ros.visualisation.trajectory_visualisation import save_trajectories_plot


class BenchmarkRun(object):
    def __init__(self, run_id, run_output_folder, benchmark_log_path, show_ros_info, headless, stage_dataset_folder, component_configuration_files, supervisor_configuration_file):

        self.benchmark_log_path = benchmark_log_path

        # components configuration parameters
        self.component_configuration_files = component_configuration_files
        self.supervisor_configuration_file = supervisor_configuration_file

        # environment parameters
        self.stage_world_folder = stage_dataset_folder
        self.stage_world_file = path.join(stage_dataset_folder, "environment.world")

        # run parameters
        self.run_id = run_id
        self.run_output_folder = run_output_folder
        self.components_ros_output = 'screen' if show_ros_info else 'log'
        self.headless = headless

        # run variables
        self.ros_has_shutdown = False

        # prepare folder structure
        run_configuration_copy_path = path.join(self.run_output_folder, "components_configuration")
        run_info_file_path = path.join(self.run_output_folder, "run_info.yaml")
        backup_file_if_exists(self.run_output_folder)
        os.mkdir(self.run_output_folder)
        os.mkdir(run_configuration_copy_path)

        # write info about the run to file
        run_info_dict = dict()
        run_info_dict["components_configuration"] = component_configuration_files
        run_info_dict["supervisor_configuration"] = supervisor_configuration_file
        run_info_dict["environment_folder"] = stage_dataset_folder
        run_info_dict["run_folder"] = self.run_output_folder
        run_info_dict["run_id"] = self.run_id
        with open(run_info_file_path, 'w') as run_info_file:
            yaml.dump(run_info_dict, run_info_file, default_flow_style=False)

        # copy the configuration to the run folder
        for component_name, configuration_path in self.component_configuration_files.items():
            configuration_copy_path = path.join(run_configuration_copy_path, "{}_{}".format(component_name, path.basename(configuration_path)))
            backup_file_if_exists(configuration_copy_path)
            shutil.copyfile(configuration_path, configuration_copy_path)

        supervisor_configuration_copy_path = path.join(run_configuration_copy_path, "{}_{}".format("supervisor", path.basename(self.supervisor_configuration_file)))
        backup_file_if_exists(supervisor_configuration_copy_path)
        shutil.copyfile(self.supervisor_configuration_file, supervisor_configuration_copy_path)

    def log(self, event):

        if not path.exists(self.benchmark_log_path):
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write("{t}, {run_id}, {event}\n".format(t="timestamp", run_id="run_id", event="event"))

        t = time.time()

        try:
            with open(self.benchmark_log_path, 'a') as output_file:
                output_file.write("{t}, {run_id}, {event}\n".format(t=t, run_id=self.run_id, event=event))
        except IOError as e:
            print_error("benchmark_log: could not write event to file: {t}, {run_id}, {event}".format(t=t, run_id=self.run_id, event=event))
            print_error(e)

    def execute_run(self):

        # components parameters
        Component.common_parameters = {'headless': self.headless, 'output': self.components_ros_output}
        environment_params = {'stage_world_file': self.stage_world_file}
        recorder_params = {'bag_file_path': path.join(self.run_output_folder, "odom_tf_ground_truth.bag")}
        slam_params = {'configuration': self.component_configuration_files['gmapping']}
        explorer_params = {'configuration': self.component_configuration_files['explore_lite']}
        navigation_params = {'configuration': self.component_configuration_files['move_base']}
        supervisor_params = {'run_output_folder': self.run_output_folder, 'configuration': self.supervisor_configuration_file}

        # declare components
        roscore = Component('roscore', 'performance_modelling', 'roscore.launch')
        rviz = Component('rviz', 'performance_modelling', 'rviz.launch')
        environment = Component('stage', 'performance_modelling', 'stage.launch', environment_params)
        recorder = Component('recorder', 'performance_modelling', 'rosbag_recorder.launch', recorder_params)
        slam = Component('gmapping', 'performance_modelling', 'gmapping.launch', slam_params)
        navigation = Component('move_base', 'performance_modelling', 'move_base.launch', navigation_params)
        explorer = Component('explore_lite', 'performance_modelling', 'explore_lite.launch', explorer_params)
        supervisor = Component('supervisor', 'performance_modelling', 'slam_benchmark_supervisor.launch', supervisor_params)

        # launch roscore and setup a node to monitor ros
        roscore.launch()
        rospy.init_node("benchmark_monitor", anonymous=True)

        # launch components
        print_info("execute_run: launching components")
        rviz.launch()
        environment.launch(headless=False)  # Override headless parameter TODO stage still does not run correctly in headless mode
        recorder.launch()
        slam.launch()
        navigation.launch()
        explorer.launch()
        supervisor.launch()

        # TODO check if all components launched properly

        # wait for the supervisor component to finish
        print_info("execute_run: waiting for supervisor to finish")
        self.log(event="waiting_supervisor_finish")
        supervisor.wait_to_finish()
        print_info("execute_run: supervisor has shutdown")
        self.log(event="supervisor_shutdown")

        if rospy.is_shutdown():
            print_error("execute_run: supervisor finished by ros_shutdown")
            self.ros_has_shutdown = True

        # shutdown remaining components
        explorer.shutdown()
        navigation.shutdown()
        slam.shutdown()
        recorder.shutdown()
        environment.shutdown()
        rviz.shutdown()
        roscore.shutdown()
        print_info("execute_run: components shutdown completed")

        self.log(event="start_compute_map_metrics")
        compute_map_metrics(self.run_output_folder, self.stage_world_folder)
        self.log(event="start_compute_localization_metrics")
        compute_localization_metrics(self.run_output_folder)
        print_info("execute_run: metrics computation completed")

        self.log(event="start_save_trajectories_plot")
        save_trajectories_plot(self.run_output_folder)
        print_info("execute_run: saved visualisation files")

        self.log(event="run_end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the benchmark')

    parser.add_argument('-e', dest='environment_dataset_folder',
                        help='Dataset folder containg the stage environment.world file (recursively).',
                        type=str,
                        default="~/ds/performance_modelling_few_datasets/",
                        required=False)

    parser.add_argument('-c', dest='benchmark_configuration',
                        help='Yaml file with the configuration of the benchmark.',
                        type=str,
                        default="~/w/catkin_ws/src/performance_modelling/config/benchmark_configurations/slam_benchmark_1.yaml",
                        required=False)

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder in which the result of each run will be placed.',
                        type=str,
                        default="~/ds/performance_modelling_output/test/",
                        required=False)

    parser.add_argument('-n', '--num-runs', dest='num_runs',
                        help='Number of runs to be executed for each combination of configurations.',
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument('-g', '--headless', dest='headless',
                        help='When set the components are run with no GUI.',
                        action='store_true',
                        required=False)

    parser.add_argument('-s', '--show-ros-info', dest='show_ros_info',
                        help='When set the component nodes are launched with output="screen".',
                        action='store_true',
                        required=False)

    args = parser.parse_args()
    base_run_folder = path.expanduser(args.base_run_folder)
    environment_dataset_folder = path.expanduser(args.environment_dataset_folder)
    benchmark_configuration = path.expanduser(args.benchmark_configuration)

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('performance_modelling')
    components_configurations_folder = path.join(package_path, "config/component_configurations/")

    if not path.exists(base_run_folder):
        os.makedirs(base_run_folder)

    log_file_path = path.join(base_run_folder, "benchmark_log.csv")

    environment_folders = sorted(map(path.dirname, set(glob.glob(path.join(path.abspath(path.expanduser(environment_dataset_folder)), "**/*.world"))).union(set(glob.glob(path.join(path.abspath(path.expanduser(environment_dataset_folder)), "*.world"))))))
    print_info("environments found: {}".format(len(environment_folders)))

    with open(benchmark_configuration, 'r') as f:
        benchmark_configuration = yaml.load(f)

    supervisor_configuration_path = path.join(components_configurations_folder, benchmark_configuration['supervisor_configuration'])

    components_configurations_dict = benchmark_configuration['components_configurations']

    # convert components configurations relative paths to absolute paths
    for component, configurations_relative_path_list in components_configurations_dict.items():
        components_configurations_dict[component] = map(lambda relative_path: path.join(components_configurations_folder, relative_path), configurations_relative_path_list)

    # convert the dict with {component_name: [configuration_1, configuration_2]} to the list [(component_name, configuration_1), (component_name, configuration_2), ...]
    configurations_alternatives = list()
    for component, configurations_list in components_configurations_dict.items():
        component_configuration_list_of_tuples = map(lambda configuration: (component, configuration), configurations_list)
        configurations_alternatives.append(component_configuration_list_of_tuples)

    # obtain the list of combinations from the list of alternatives
    # example: component 1 with configurations [A, B, C, D], and component 2 with configurations [x, y]:
    #   itertools.product([A, B, C, D], [x, y]) --> [[A, x], [A, y], [B, x], [B, y], [C, x], [C, y], [D, x], [D, y]]
    configuration_combinations_lists = list(itertools.product(*configurations_alternatives))

    # convert the list of lists to a list of dicts
    configuration_combinations_dicts = map(dict, configuration_combinations_lists)

    num_combinations = len(configuration_combinations_dicts)
    print_info("number of configuration combinations: {}".format(num_combinations))
    print_info("number of runs per combination: {}".format(args.num_runs))
    print_info("total number of runs: {}".format(args.num_runs * num_combinations))

    for _ in range(args.num_runs):
        for environment_folder in environment_folders:
            for components_configurations in configuration_combinations_dicts:

                # find an available run folder path
                i = 0
                run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))
                while path.exists(run_folder):
                    i += 1
                    run_folder = path.join(base_run_folder, "run_{run_number}".format(run_number=i))

                print_info("benchmark: starting run {run_index}".format(run_index=i))
                print_info("\tenvironment_folder:", environment_folder)
                print_info("\tsupervisor_configuration:", supervisor_configuration_path)
                print_info("\tcomponents_configurations:")
                for k, v in components_configurations.items():
                    print_info("\t\t{}: ...{}".format(k, v[-100:]))

                # instantiate and execute the run
                try:
                    r = BenchmarkRun(run_id=i,
                                     run_output_folder=run_folder,
                                     benchmark_log_path=log_file_path,
                                     show_ros_info=args.show_ros_info,
                                     headless=args.headless,
                                     stage_dataset_folder=environment_folder,
                                     component_configuration_files=components_configurations,
                                     supervisor_configuration_file=supervisor_configuration_path)

                    r.execute_run()

                    if r.ros_has_shutdown:
                        print_info("benchmark: run {run_index} interrupted".format(run_index=i))
                        print_info("benchmark: interrupted")
                        sys.exit(0)
                    else:
                        print_info("benchmark: run {run_index} completed".format(run_index=i))

                except roslaunch.RLException:
                    print_error(traceback.format_exc())
                    sys.exit(0)
                except IOError:
                    print_error(traceback.format_exc())
                except ValueError:
                    print_error(traceback.format_exc())

    print_info("benchmark: finished")
