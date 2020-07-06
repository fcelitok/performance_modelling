#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
import subprocess
from os import path

dataset_path = path.expanduser("~/ds/performance_modelling/test_datasets/dataset")

compressed_files = glob.glob(dataset_path + '/**/*.tar.xz', recursive=True)

for file_path in compressed_files:
    file_size = path.getsize(file_path)
    print('\n', file_size // 1024, 'KiB', file_path)
    file_dir = path.dirname(file_path)
    # file_name = path.basename(file_path)
    cmd = ['tar', '-xJf', file_path]
    print(f"executing {' '.join(cmd)} in {file_dir}")
    subprocess.call(cmd, cwd=file_dir)
