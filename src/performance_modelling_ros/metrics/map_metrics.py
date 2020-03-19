#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import numpy as np
import os
import yaml
from PIL import Image
from os import path

from performance_modelling_ros.utils import print_info, print_error, backup_file_if_exists


def compute_map_metrics(run_output_folder, stage_world_folder=None):
    print_metric_results = False
    map_snapshots_path = path.join(run_output_folder, "benchmark_data", "map_snapshots")
    last_map_snapshot_path = path.join(map_snapshots_path, "last_map.pgm")
    last_map_info_path = path.join(map_snapshots_path, "last_map_info.yaml")

    if stage_world_folder is None:
        with open(path.join(run_output_folder, "run_info.yaml"), 'r') as run_info_file:
            stage_world_folder = yaml.load(run_info_file)['environment_folder']

    ground_truth_map_info_file_path = path.join(stage_world_folder, "stage_world_info.yaml")
    ground_truth_map_file_path = path.join(stage_world_folder, "map_ground_truth.pgm")

    metric_results_path = path.join(run_output_folder, "metric_results")
    map_metric_result_file_path = path.join(metric_results_path, "normalised_explored_area.yaml")

    # create folders structure
    if not path.exists(metric_results_path):
        os.makedirs(metric_results_path)

    # check required files exist
    if not path.isfile(last_map_snapshot_path):
        print_error("compute_map_metrics: last_map_snapshot file not found {}".format(last_map_snapshot_path))
        return

    if not path.isfile(last_map_info_path):
        print_error("compute_map_metrics: last_map_info file not found {}".format(last_map_info_path))
        return

    if not path.isfile(ground_truth_map_info_file_path):
        print_error("compute_map_metrics: stage_map_info_file file not found {}".format(ground_truth_map_info_file_path))
        return

    with open(ground_truth_map_info_file_path, 'r') as stage_info_file:
        stage_info_yaml = yaml.load(stage_info_file)

    ground_truth_map = Image.open(ground_truth_map_file_path)

    if print_metric_results:
        print("opened ground_truth_map, mode: {mode}, image: {im}".format(mode=ground_truth_map.mode, im=ground_truth_map_file_path))

    ground_truth_map_size_meters = np.array([float(stage_info_yaml['map']['size']['x']), float(stage_info_yaml['map']['size']['y'])])
    ground_truth_map_size_pixels = np.array(ground_truth_map.size)
    ground_truth_resolution = ground_truth_map_size_meters / ground_truth_map_size_pixels  # meter/pixel, on both axis, except y axis is inverted in image
    ground_truth_cell_area = float(ground_truth_resolution[0] * ground_truth_resolution[1])  # width × height of one pixel, meters^2

    ground_truth_free_cell_count = 0
    ground_truth_occupied_cell_count = 0
    ground_truth_unknown_cell_count = 0
    ground_truth_total_cell_count = ground_truth_map.size[0] * ground_truth_map.size[1]
    ground_truth_pixels = ground_truth_map.load()
    for i in range(ground_truth_map.size[0]):
        for j in range(ground_truth_map.size[1]):
            ground_truth_free_cell_count += ground_truth_pixels[i, j] == (254, 254, 254)
            ground_truth_occupied_cell_count += ground_truth_pixels[i, j] == (0, 0, 0)
            ground_truth_unknown_cell_count += ground_truth_pixels[i, j] == (205, 205, 205)

    explorable_area = ground_truth_free_cell_count * ground_truth_cell_area

    result_map = Image.open(last_map_snapshot_path)

    if print_metric_results:
        print("opened result_map, mode: {mode}, image: {im}".format(mode=result_map.mode, im=last_map_snapshot_path))

    with open(last_map_info_path, 'r') as result_map_info_file:
        result_map_info_yaml = yaml.load(result_map_info_file)

    ground_truth_map_size_pixels = np.array(result_map.size)
    result_map_resolution = float(result_map_info_yaml['info']['resolution'])  # meter/pixel, on both axis
    result_cell_area = result_map_resolution**2  # length of one pixel squared, meters^2

    result_free_cell_count = 0
    result_occupied_cell_count = 0
    result_unknown_cell_count = 0
    result_total_cell_count = result_map.size[0] * result_map.size[1]
    result_map_pixels = result_map.load()
    for i in range(result_map.size[0]):
        for j in range(result_map.size[1]):
            result_free_cell_count += result_map_pixels[i, j] == 254
            result_occupied_cell_count += result_map_pixels[i, j] == 0
            result_unknown_cell_count += result_map_pixels[i, j] == 205

    explored_area = result_free_cell_count * result_cell_area

    if print_metric_results:
        print_info("normalised explored area:", explored_area / explorable_area)

    yaml_dict = {
        'result_map': {
            'count': {
                'free': result_free_cell_count,
                'occupied': result_occupied_cell_count,
                'unknown': result_unknown_cell_count,
                'total': result_total_cell_count,
            },
            'area': {
                'free': explored_area,
                'occupied': result_occupied_cell_count * result_cell_area,
                'unknown': result_unknown_cell_count * result_cell_area,
                'total': result_total_cell_count * result_cell_area,
            },
        },
        'ground_truth_map': {
            'count': {
                'free': ground_truth_free_cell_count,
                'occupied': ground_truth_occupied_cell_count,
                'unknown': ground_truth_unknown_cell_count,
                'total': ground_truth_total_cell_count,
            },
            'area': {
                'free': explorable_area,
                'occupied': ground_truth_occupied_cell_count * ground_truth_cell_area,
                'unknown': ground_truth_unknown_cell_count * ground_truth_cell_area,
                'total': ground_truth_total_cell_count * ground_truth_cell_area,
            },
        },
        'normalised_explored_area': float(explored_area / explorable_area)
    }
    backup_file_if_exists(map_metric_result_file_path)
    with open(map_metric_result_file_path, 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file, default_flow_style=False)


if __name__ == '__main__':
    run_folders = filter(path.isdir, glob.glob(path.expanduser("~/ds/performance_modelling_output/test/*")))
    last_run_folder = sorted(run_folders, key=lambda x: path.getmtime(x))[-1]
    print("last run folder:", last_run_folder)
    compute_map_metrics(last_run_folder)
