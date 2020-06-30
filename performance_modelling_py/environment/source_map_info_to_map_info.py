#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from collections import OrderedDict
from os import path
import numpy as np
import yaml
from PIL import Image
from performance_modelling_py.environment.ground_truth_map_utils import GroundTruthMap, trim

environment_paths = sorted(glob.glob(path.expanduser("~/ds/performance_modelling/dataset_v3/*")))


def get_trimmed_image(input_map_path, occupied_threshold=150):

    input_image = Image.open(input_map_path)

    if input_image.mode != 'RGB':
        # remove alpha channel by pasting on white background
        background = Image.new("RGB", input_image.size, GroundTruthMap.free_rgb)
        background.paste(input_image)
        input_image = background

    # saturate all colors below the threshold to black and all values above threshold to the unknown color value (they will be colored to free later)
    # this is needed because some images have been heavily compressed and contain all sort of colors
    threshold_image = input_image.point(lambda original_value: GroundTruthMap.occupied if original_value < occupied_threshold else GroundTruthMap.unknown)
    threshold_image.save(path.expanduser("~/tmp/threshold_image.pgm"))

    # trim borders not containing black pixels (some simulators ignore non-black borders while placing the pixels in the simulated map and computing its resolution, so they need to be ignored in the following calculations)
    trimmed_image = trim(threshold_image, GroundTruthMap.unknown_rgb)
    trimmed_image.save(path.expanduser("~/tmp/trimmed_image.pgm"))

    return trimmed_image


def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)


for environment_path in environment_paths:
    environment_name = path.basename(environment_path)
    data_path = path.join(environment_path, "data")
    source_map_image_path = path.join(data_path, "source_map.png")
    source_map_info_file_path = path.join(data_path, "source_map_info.yaml")
    map_info_file_path = path.join(data_path, "map.yaml")

    print(f"\n\n--- {environment_name}\n")

    with open(source_map_info_file_path) as source_map_info_file:
        source_map_info = yaml.load(source_map_info_file)

    source_image = get_trimmed_image(source_map_image_path)

    map_size_pixels = np.array(source_image.size)
    print("map_size_pixels", map_size_pixels)

    print()

    source_map_size_meters = np.array([source_map_info['map']['size']['x'], source_map_info['map']['size']['y']], dtype=float)
    print("source_map_size_meters", source_map_size_meters)

    source_resolution = source_map_size_meters / map_size_pixels
    print("source_resolution", source_resolution)

    source_resolution_avg = np.sum(source_resolution) / 2
    print("r_avg", source_resolution_avg)

    source_initial_position_map_frame = np.array([source_map_info['robot']['pose']['x'], source_map_info['robot']['pose']['y']], dtype=float)
    print("source_initial_position_map_frame", source_initial_position_map_frame)

    source_map_offset_meters = np.array([source_map_info['map']['pose']['x'], source_map_info['map']['pose']['y']], dtype=float)
    print("source_map_offset_meters", source_map_offset_meters)

    source_map_frame_from_bottom_left = source_map_size_meters / 2 - source_map_offset_meters
    print("source_map_frame_from_bottom_left", source_map_frame_from_bottom_left)

    print()

    norm_map_frame_from_bottom_left = (source_map_size_meters / 2 - source_map_offset_meters) / source_map_size_meters
    print("norm_map_frame_from_bottom_left", norm_map_frame_from_bottom_left)

    norm_initial_position_map_frame = source_initial_position_map_frame / source_map_size_meters
    print("norm_initial_position_map_frame", norm_initial_position_map_frame)

    print()

    new_resolution = max(0.01, np.around(source_resolution_avg / 0.01) * 0.01)
    print("r_avg/0.01", source_resolution_avg / 0.01)
    print("np.around(r_avg/0.01)", np.around(source_resolution_avg / 0.01))
    print("new_resolution", new_resolution)

    new_map_size_meters = new_resolution * map_size_pixels
    print("new_map_size_meters", new_map_size_meters)

    new_map_frame_from_bottom_left = norm_map_frame_from_bottom_left * new_map_size_meters
    print("new_map_frame_from_bottom_left", new_map_frame_from_bottom_left)

    new_map_offset = -new_map_frame_from_bottom_left
    print("new_map_offset", new_map_offset)

    new_initial_position_map_frame = norm_initial_position_map_frame * new_map_size_meters
    print("new_initial_position_map_frame", new_initial_position_map_frame)

    map_info = OrderedDict()
    map_info['image'] = "map.pgm"
    map_info['resolution'] = float(new_resolution)
    map_info['origin'] = list(map(float, new_map_offset)) + [0.0]
    map_info['initial_pose'] = list(map(float, new_initial_position_map_frame))
    map_info['negate'] = 0
    map_info['occupied_thresh'] = 0.65
    map_info['free_thresh'] = 0.196

    setup_yaml()
    print("\nmap.yaml")
    print(yaml.dump(map_info))

    with open(map_info_file_path, 'w') as map_info_file:
        yaml.dump(map_info, map_info_file)
