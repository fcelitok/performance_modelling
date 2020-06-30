import glob
import os
import shutil
from os import path

source_dataset_path = path.expanduser("~/ds/performance_modelling/dataset_raw")
new_dataset_path = path.expanduser("~/ds/performance_modelling/dataset_new")
gazebo_template_path = path.expanduser("~/ds/performance_modelling_test_datasets/gazebo_template/gazebo")

if not path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

source_environment_paths = glob.glob(source_dataset_path + '/*')

for source_environment_path in source_environment_paths:
    environment_name = path.basename(source_environment_path)
    new_environment_path = path.join(new_dataset_path, environment_name)
    print(environment_name)
    print(source_environment_path)
    print(new_environment_path)

    os.mkdir(new_environment_path)
    os.mkdir(path.join(new_environment_path, "data"))
    os.mkdir(path.join(new_environment_path, "gazebo"))
    os.mkdir(path.join(new_environment_path, "gazebo/model"))
    os.mkdir(path.join(new_environment_path, "stage"))

    shutil.copy(path.join(source_environment_path, "data/map.png"),
                path.join(new_environment_path, "data/source_map.png"))
    os.chmod(path.join(new_environment_path, "data/source_map.png"), 0o644)

    shutil.copy(path.join(source_environment_path, "data/map_info.yaml"),
                path.join(new_environment_path, "data/source_map_info.yaml"))
    os.chmod(path.join(new_environment_path, "data/source_map_info.yaml"), 0o644)

    for source_stage_file_path in glob.glob(path.join(source_environment_path, "stage") + "/*"):
        new_stage_file_path = path.join(new_environment_path, "stage", path.basename(source_stage_file_path))
        shutil.copy(source_stage_file_path, new_stage_file_path)
        os.chmod(new_stage_file_path, 0o644)

    gazebo_file_relative_paths = [
        "model/model.config",
        "model/model.sdf",
        "gazebo_environment.model",
        "robot.urdf",
    ]
    for gazebo_file_relative_path in gazebo_file_relative_paths:
        source_gazebo_file_path = path.join(gazebo_template_path, gazebo_file_relative_path)
        new_gazebo_file_path = path.join(new_environment_path, "gazebo", gazebo_file_relative_path)
        with open(source_gazebo_file_path) as sf:
            gazebo_s = sf.read().replace("{{environment_name}}", environment_name)
        with open(new_gazebo_file_path, 'w') as nf:
            nf.write(gazebo_s)
