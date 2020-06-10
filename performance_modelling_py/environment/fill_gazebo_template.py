import glob
import os
from os import path

target_dataset_path = path.expanduser("~/ds/performance_modelling/test_datasets/dataset")
gazebo_template_path = path.expanduser("~/ds/performance_modelling/test_datasets/gazebo_template/gazebo")
source_environment_paths = filter(path.isdir, glob.glob(target_dataset_path + '/*'))

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
