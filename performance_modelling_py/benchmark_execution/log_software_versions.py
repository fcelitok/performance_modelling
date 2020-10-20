#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import glob
import os
import sys
from os import path
import subprocess

from performance_modelling_py.utils import print_error, backup_file_if_exists, print_info


def log_packages_and_repos(source_workspace_path, log_dir_path):
    source_workspace_path = path.normpath(path.expanduser(source_workspace_path))
    log_dir_path = path.normpath(path.expanduser(log_dir_path))

    if path.isdir(log_dir_path):
        backup_file_if_exists(log_dir_path)
        os.makedirs(log_dir_path)
    else:
        if not path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        else:
            print_error("log_dir_path already exists but is not a directory [{}]".format(log_dir_path))
            return False

    source_repos_version_file_path = path.join(log_dir_path, "source_repos_version")
    source_repos_status_file_path = path.join(log_dir_path, "source_repos_status")
    installed_packages_version_file_path = path.join(log_dir_path, "installed_packages_version")
    ros_packages_file_path = path.join(log_dir_path, "ros_packages")

    if not path.isdir(source_workspace_path):
        print_error("source_workspace_path does not exists or is not a directory [{}]".format(source_workspace_path))
        return False

    with open(source_repos_version_file_path, 'w') as source_repos_version_list_file:
        source_repo_git_path_list = glob.glob(path.join(source_workspace_path, "**", ".git"))
        for source_repo_git_path in source_repo_git_path_list:
            source_repo_path = path.abspath(path.join(source_repo_git_path, ".."))
            head_hash_output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_repo_path)
            head_hash = head_hash_output.rstrip().replace('\n', ';')
            source_repos_version_list_file.write("{}, {}\n".format(source_repo_path, head_hash))

    with open(source_repos_status_file_path, 'w') as source_repos_status_list_file:
        source_repo_git_path_list = glob.glob(path.join(source_workspace_path, "**", ".git"))
        for source_repo_git_path in source_repo_git_path_list:
            source_repo_path = path.abspath(path.join(source_repo_git_path, ".."))
            status_output = subprocess.check_output(["git", "status"], cwd=source_repo_path)
            diff_output = subprocess.check_output(["git", "diff"], cwd=source_repo_path)
            source_repos_status_list_file.write("STATUS_BEGIN {repo_path}\n{status}STATUS_END\nDIFF_BEGIN {repo_path}\n{diff}\nDIFF_END\n".format(repo_path=source_repo_path, status=status_output, diff=diff_output))

    with open(installed_packages_version_file_path, 'w') as installed_packages_version_file:
        dpkg_list_output = subprocess.check_output(["dpkg", "-l"])
        installed_packages_version_file.write(dpkg_list_output)

    with open(ros_packages_file_path, 'w') as ros_packages_file:
        rospack_list_output = subprocess.check_output(["rospack", "list"], cwd=source_workspace_path)
        ros_packages_file.write(rospack_list_output)

    return True


if __name__ == '__main__':
    default_source_workspace_path = "~/w/catkin_ws/src/"
    default_base_run_folder = "~/ds/performance_modelling/output/test_slam/"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Add software version information to all run output folders.')

    parser.add_argument('-r', dest='base_run_folder',
                        help='Folder in which the result of each run will be placed. Defaults to {}.'.format(default_base_run_folder),
                        type=str,
                        default=default_base_run_folder,
                        required=False)

    parser.add_argument('-s', dest='source_workspace_path',
                        help='Path of the workspace directory. Defaults to {}.'.format(default_source_workspace_path),
                        type=str,
                        default=default_source_workspace_path,
                        required=False)

    args = parser.parse_args()
    base_run_folder_arg = path.expanduser(args.base_run_folder)
    source_workspace_path_arg = path.expanduser(args.source_workspace_path)

    run_folders = list(filter(path.isdir, glob.glob(path.join(base_run_folder_arg, "*"))))
    for progress, run_folder in enumerate(run_folders):
        print_info("main: log_packages_and_repos {}% {}".format((progress + 1) * 100 / len(run_folders), run_folder))
        run_folder_log_dir_path = path.join(run_folder, "software_versions_log")
        log_packages_and_repos(source_workspace_path=source_workspace_path_arg, log_dir_path=run_folder_log_dir_path)
