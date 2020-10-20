#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import argparse
import shutil

import yaml
from os import path

from performance_modelling_py.utils import print_info, print_error


def get_yaml(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        yaml_obj = yaml.load(yaml_file)
    return yaml_obj


def update_run_parameters(base_run_folder_path):

    base_run_folder = path.expanduser(base_run_folder_path)

    if not path.isdir(base_run_folder):
        print_error("update_run_parameters: base_run_folder does not exists or is not a directory".format(base_run_folder))
        return None, None

    run_folders = sorted(list(filter(path.isdir, glob.glob(path.abspath(base_run_folder) + '/*'))))

    # collect results from runs not already cached
    print_info("update_run_parameters: reading run data")
    no_output = True
    for i, run_folder in enumerate(run_folders):
        current_run_info_file_path = path.join(run_folder, "run_info.yaml")
        original_run_info_file_path = path.join(run_folder, "run_info_original.yaml")

        if not path.exists(current_run_info_file_path) and not path.exists(original_run_info_file_path):
            print_error("update_run_parameters: run_info file does not exists [{}]".format(run_folder))
            no_output = False
            continue

        # backup current run_info if there is not already a backup of the original
        # do not overwrite the original, since it will always be used to create the updated run_info
        if not path.exists(original_run_info_file_path):
            shutil.copyfile(current_run_info_file_path, original_run_info_file_path)

        with open(original_run_info_file_path) as original_run_info_file:
            run_info = yaml.load(original_run_info_file)

        if 'run_parameters' not in run_info:
            print_error("update_run_parameters: run_parameters not in run_info [{}]".format(run_folder))
            no_output = False
            continue

        if 'slam_node' not in run_info['run_parameters']:
            print_error("slam_node not in run_info['run_parameters'] [{}]".format(run_folder))
            no_output = False
            continue

        slam_node = run_info['run_parameters']['slam_node']

        # linear_update, angular_update -> linear_angular_update
        if slam_node == 'slam_toolbox':
            slam_config_path = path.join(run_folder, "components_configuration", "slam_toolbox", "slam_toolbox_online_async.yaml")
            if not path.exists(slam_config_path):
                print_error("not path.exists(slam_config_path) [{}]".format(slam_config_path))
                no_output = False
                continue
            slam_config = get_yaml(slam_config_path)
            linear_update = slam_config['minimum_travel_distance']
            angular_update = slam_config['minimum_travel_heading']
            run_info['run_parameters']['linear_angular_update'] = [linear_update, angular_update]
        elif slam_node == 'gmapping':
            slam_config_path = path.join(run_folder, "components_configuration", "gmapping", "gmapping.yaml")
            if not path.exists(slam_config_path):
                print_error("not path.exists(slam_config_path) [{}]".format(slam_config_path))
                no_output = False
                continue
            slam_config = get_yaml(slam_config_path)
            linear_update = slam_config['linearUpdate']
            angular_update = slam_config['angularUpdate']
            run_info['run_parameters']['linear_angular_update'] = [linear_update, angular_update]
        else:
            print_error("slam_node = {}".format(slam_node))
            no_output = False
            continue

        if 'linear_update' in run_info['run_parameters']:
            del run_info['run_parameters']['linear_update']
        if 'angular_update' in run_info['run_parameters']:
            del run_info['run_parameters']['angular_update']

        # remove lidar_model
        if 'lidar_model' in run_info['run_parameters']:
            del run_info['run_parameters']['lidar_model']

        # add goal_tolerance
        nav_config_path = path.join(run_folder, "components_configuration", "move_base", "move_base_tb3.yaml")
        if not path.exists(nav_config_path):
            print_error("not path.exists(nav_config_path) [{}]".format(nav_config_path))
            no_output = False
            continue
        nav_config = get_yaml(nav_config_path)
        xy_goal_tolerance = nav_config['DWAPlannerROS']['xy_goal_tolerance']
        yaw_goal_tolerance = nav_config['DWAPlannerROS']['yaw_goal_tolerance']
        run_info['run_parameters']['goal_tolerance'] = [xy_goal_tolerance, yaw_goal_tolerance]

        # add fewer_nav_goals (false if not specified in the configuration)
        sup_config_path = path.join(run_folder, "components_configuration", "slam_benchmark_supervisor", "slam_benchmark_supervisor.yaml")
        if not path.exists(sup_config_path):
            print_error("not path.exists(sup_config_path) [{}]".format(sup_config_path))
            no_output = False
            continue
        sup_config = get_yaml(sup_config_path)
        if 'fewer_nav_goals' in sup_config:
            fewer_nav_goals = sup_config['fewer_nav_goals']
        else:
            fewer_nav_goals = False
        run_info['run_parameters']['fewer_nav_goals'] = fewer_nav_goals

        with open(current_run_info_file_path, 'w') as current_run_info_file:
            yaml.dump(run_info, current_run_info_file, default_flow_style=False)

        print_info("update_run_parameters: {}%".format(int((i + 1)*100/len(run_folders))), replace_previous_line=no_output)
        no_output = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Execute the analysis of the benchmark results.')

    default_base_run_folder = "~/ds/performance_modelling/output/test_slam/"
    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder containing the result the runs. Defaults to {}'.format(default_base_run_folder),
                        type=str,
                        default=default_base_run_folder,
                        required=False)

    args = parser.parse_args()
    update_run_parameters(args.base_run_folder)
