#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
import subprocess
from os import path

dataset_path = path.expanduser("~/ds/performance_modelling/test_datasets/dataset")

dataset_files = glob.glob(dataset_path + '/**/*.dae', recursive=True) + \
                glob.glob(dataset_path + '/**/*.pgm', recursive=True) + \
                glob.glob(dataset_path + '/**/*.png', recursive=True) + \
                glob.glob(dataset_path + '/**/*.svg', recursive=True) + \
                glob.glob(dataset_path + '/**/*.posegraph', recursive=True) + \
                glob.glob(dataset_path + '/**/*.data', recursive=True)

for file_path in dataset_files:
    file_size = path.getsize(file_path)
    print(file_size // 1024, 'KiB', file_path)
    file_dir = path.dirname(file_path)
    file_name = path.basename(file_path)
    cmd = ['tar', '-cJf', file_name + '.tar.xz', file_name]
    print(f"executing {' '.join(cmd)} in {file_dir}")
    subprocess.call(cmd, cwd=file_dir)
