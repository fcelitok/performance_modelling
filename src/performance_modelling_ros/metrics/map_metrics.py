#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import os
import yaml
from PIL import Image
from os import path

from performance_modelling_ros.utils import print_info, print_error, backup_file_if_exists


def compute_map_metrics(run_output_folder, stage_world_folder):
    print_metric_results = False
    metric_results_path = path.join(run_output_folder, "metric_results")
    log_files_path = path.join(run_output_folder, "logs")
    map_snapshots_path = path.join(run_output_folder, "map_snapshots")
    last_map_snapshot_path = path.join(map_snapshots_path, "last_map.pgm")
    last_map_info_path = path.join(map_snapshots_path, "last_map_info.yaml")

    ground_truth_map_info_file_path = path.join(stage_world_folder, "stage_world_info.yaml")
    ground_truth_map_file_path = path.join(stage_world_folder, "map_ground_truth.pgm")

    map_metric_result_file_path = path.join(metric_results_path, "normalised_explored_area.yaml")

    # create folders structure
    if not path.exists(metric_results_path):
        os.makedirs(metric_results_path)

    if not path.exists(log_files_path):
        os.makedirs(log_files_path)

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
    ground_truth_cell_area = ground_truth_resolution[0] * ground_truth_resolution[1]  # width × height of one pixel, meters^2

    if print_metric_results:
        print("ground truth map size:", ground_truth_map_size_meters, "m")
        print("ground truth map size:", ground_truth_map_size_pixels, "pixels")
        print("ground truth map resolution:", ground_truth_resolution, "m")
        print("ground truth map cell area:", ground_truth_cell_area, "m²")

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

    if print_metric_results:
        print("ground truth:")
        print("\tcount:")
        print("\t\tfree:", ground_truth_free_cell_count, ground_truth_free_cell_count/float(ground_truth_total_cell_count))
        print("\t\toccupied:", ground_truth_occupied_cell_count, ground_truth_occupied_cell_count/float(ground_truth_total_cell_count))
        print("\t\tunknown:", ground_truth_unknown_cell_count, ground_truth_unknown_cell_count/float(ground_truth_total_cell_count))
        print("\t\ttotal:", ground_truth_total_cell_count)

        print("\tarea:")
        print("\t\tfree:", explorable_area, "m²")
        print("\t\toccupied:", ground_truth_occupied_cell_count * ground_truth_cell_area, "m²")
        print("\t\tunknown:", ground_truth_unknown_cell_count * ground_truth_cell_area, "m²")
        print("\t\ttotal:", ground_truth_total_cell_count * ground_truth_cell_area, "m²")

    result_map = Image.open(last_map_snapshot_path)

    if print_metric_results:
        print("opened result_map, mode: {mode}, image: {im}".format(mode=result_map.mode, im=last_map_snapshot_path))

    with open(last_map_info_path, 'r') as result_map_info_file:
        result_map_info_yaml = yaml.load(result_map_info_file)

    ground_truth_map_size_pixels = np.array(result_map.size)
    result_map_resolution = float(result_map_info_yaml['info']['resolution'])  # meter/pixel, on both axis
    result_cell_area = result_map_resolution**2  # length of one pixel squared, meters^2

    if print_metric_results:
        print("map_size:", ground_truth_map_size_pixels, "pixels")
        print("result map resolution:", result_map_resolution, "m")
        print("result map cell area:", result_cell_area, "m²")

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
        print("result map:")
        print("\tcount:")
        print("\t\tfree:", result_free_cell_count, result_free_cell_count/float(result_total_cell_count))
        print("\t\toccupied:", result_occupied_cell_count, result_occupied_cell_count/float(result_total_cell_count))
        print("\t\tunknown:", result_unknown_cell_count, result_unknown_cell_count/float(result_total_cell_count))
        print("\t\ttotal:", result_total_cell_count)

        print("\tarea:")
        print("\t\tfree:", explored_area, "m²")
        print("\t\toccupied:", result_occupied_cell_count * result_cell_area, "m²")
        print("\t\tunknown:", result_unknown_cell_count * result_cell_area, "m²")
        print("\t\ttotal:", result_total_cell_count * result_cell_area, "m²")

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
    compute_map_metrics(run_output_folder="/home/enrico/ds/performance_modelling_output/test/run_23", stage_world_folder="/home/enrico/ds/performance_modelling_all_datasets/test")
