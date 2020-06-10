import glob
import os
import subprocess
from os import path

dataset_path = path.expanduser("~/ds/performance_modelling/test_datasets/dataset")

dataset_files = glob.glob(dataset_path + '/**/*.dae', recursive=True) + \
                glob.glob(dataset_path + '/**/*.pgm', recursive=True) + \
                glob.glob(dataset_path + '/**/*.png', recursive=True) + \
                glob.glob(dataset_path + '/**/*.svg', recursive=True)

for file_path in dataset_files:
    file_size = path.getsize(file_path)
    print(file_size // 1024, 'KiB', file_path)
    subprocess.call(['tar', '-cJf', file_path + '.tar.xz', file_path])


