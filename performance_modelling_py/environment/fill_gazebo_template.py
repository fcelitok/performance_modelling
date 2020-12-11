#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
from os import path

from performance_modelling_py.utils import print_error

if __name__ == '__main__':
    default_target_dataset_path = "~/ds/performance_modelling/test_datasets/dataset/*"
    default_gazebo_template_path = "~/ds/performance_modelling/test_datasets/gazebo_template/gazebo"

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Copies and fills the gazebo files from template to environment folders (overwrites existing files).')
    parser.add_argument('-e', dest='target_dataset_path',
                        help='Folder containing the target dataset, wildcards are allowed to select multiple folders. Example: {}'.format(default_target_dataset_path),
                        type=str,
                        required=True)

    parser.add_argument('-t', dest='gazebo_template_path',
                        help='Folder containing the gazebo template. Example: {}'.format(default_gazebo_template_path),
                        type=str,
                        required=True)

    args = parser.parse_args()

    gazebo_template_path = path.expanduser(args.gazebo_template_path)
    target_dataset_path = path.expanduser(args.target_dataset_path)
    source_environment_paths = filter(path.isdir, glob.glob(target_dataset_path))

    if not path.exists(gazebo_template_path):
        print_error("gazebo_template_path does not exists:", gazebo_template_path)
        sys.exit(-1)

    for target_environment_path in source_environment_paths:
        environment_name = path.basename(target_environment_path)
        print("environment_name", environment_name)
        print("target_environment_path", target_environment_path)

        gazebo_file_relative_paths = map(lambda ap: path.relpath(ap, gazebo_template_path), filter(path.isfile, glob.glob(gazebo_template_path + '/**', recursive=True)))

        for gazebo_file_relative_path in gazebo_file_relative_paths:
            source_gazebo_file_path = path.join(gazebo_template_path, gazebo_file_relative_path)
            target_gazebo_file_path = path.join(target_environment_path, "gazebo", gazebo_file_relative_path)
            print("target_gazebo_file_path", target_gazebo_file_path)
            print("source_gazebo_file_path", source_gazebo_file_path)
            if not path.exists(path.dirname(target_gazebo_file_path)):
                os.makedirs(path.dirname(target_gazebo_file_path))
            with open(source_gazebo_file_path) as source_file:
                gazebo_s = source_file.read().replace("{{environment_name}}", environment_name)
            with open(target_gazebo_file_path, 'w') as target_file:
                target_file.write(gazebo_s)
