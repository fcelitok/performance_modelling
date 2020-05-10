#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
import traceback
from os import path
from PIL.ImageDraw import floodfill
from PIL import Image, ImageFilter

import numpy as np
import yaml
from performance_modelling_py.environment.gridmap_utils import color_diff, color_abs_diff, rgb_less_than

from performance_modelling_py.utils import print_info, backup_file_if_exists, print_error, print_fatal


def compute_ground_truth_from_grid_map(grid_map_file_path, grid_map_info_file_path, ground_truth_map_file_path, unknown=205, occupied_threshold=150,  blur_filter_radius=0, do_not_recompute=False, backup_if_exists=False, ground_truth_map_files_dump_path=None):
    unknown_rgb = (unknown, unknown, unknown)  # color value of unknown cells in the ground truth map
    occupied_threshold_rgb = (occupied_threshold, occupied_threshold, occupied_threshold)  # color values in the input map (black/white) less than this value are considered occupied

    if path.exists(ground_truth_map_file_path) and do_not_recompute:
        print_info("do_not_recompute: will not recompute the output image")
        return

    gt = Image.open(grid_map_file_path)

    if gt.mode != 'RGB':
        # remove alpha channel by pasting on white background
        background = Image.new("RGB", gt.size, (254, 254, 254))
        background.paste(gt)
        # apply a blur filter to reduce artifacts in images that have been saved to lossy formats
        gt = background.filter(ImageFilter.BoxBlur(blur_filter_radius))

    pixels = gt.load()

    # crop borders not containing black pixels (some simulators ignore non-black borders while placing the pixels in the simulated map and computing its resolution, so they need to be ignored in the following calculations)
    w, h = gt.size
    top_border = 0
    for y in range(h):
        found = False
        for x in range(w):
            if color_diff(pixels[x, y], (0, 0, 0)) == 0:
                top_border = y
                found = True
                break
        if found:
            break
    bottom_border = h
    for y in range(h)[::-1]:
        found = False
        for x in range(w):
            if color_diff(pixels[x, y], (0, 0, 0)) == 0:
                bottom_border = y + 1
                found = True
                break
        if found:
            break
    left_border = 0
    for x in range(w):
        found = False
        for y in range(h):
            if color_diff(pixels[x, y], (0, 0, 0)) == 0:
                left_border = x
                found = True
                break
        if found:
            break
    right_border = w
    for x in range(w)[::-1]:
        found = False
        for y in range(h):
            if color_diff(pixels[x, y], (0, 0, 0)) == 0:
                right_border = x + 1
                found = True
                break
        if found:
            break

    gt = gt.crop(box=(left_border, top_border, right_border, bottom_border))
    pixels = gt.load()

    # convert all free pixels to unknown pixels
    for i in range(gt.size[0]):
        for j in range(gt.size[1]):
            if rgb_less_than(pixels[i, j], occupied_threshold_rgb):
                pixels[i, j] = (0, 0, 0)
            else:  # not black -> unknown space
                pixels[i, j] = unknown_rgb

    with open(grid_map_info_file_path, 'r') as info_file:
        info_yaml = yaml.load(info_file)

    if info_yaml['map']['pose']['x'] != 0 or info_yaml['map']['pose']['y'] != 0 or info_yaml['map']['pose']['z'] != 0 or info_yaml['map']['pose']['theta'] != 0:
        print_error("convert_grid_map_to_gt_map: map not in origin")

    initial_position_meters = np.array([float(info_yaml['robot']['pose']['x']), float(info_yaml['robot']['pose']['y'])])
    map_size_meters = np.array([float(info_yaml['map']['size']['x']), float(info_yaml['map']['size']['y'])])
    map_size_pixels = np.array(gt.size)
    resolution = map_size_meters / map_size_pixels * np.array([1, -1])  # meter/pixel, on both axis, except y axis is inverted in image

    map_center_pixels = map_size_pixels / 2
    p_x, p_y = initial_position_pixels = list(map(int, map_center_pixels + initial_position_meters / resolution))

    if pixels[p_x, p_y] != unknown_rgb:
        print_fatal("initial position in a wall pixel")
        return

    # convert to free the pixels accessible from the initial pose
    floodfill(gt, initial_position_pixels, (254, 254, 254), thresh=10)

    if backup_if_exists:
        backup_file_if_exists(ground_truth_map_file_path)

    try:
        gt.save(ground_truth_map_file_path)
        if ground_truth_map_files_dump_path is not None:
            if not path.exists(ground_truth_map_files_dump_path):
                os.makedirs(path.basename(ground_truth_map_files_dump_path))
            gt.save(path.join(ground_truth_map_files_dump_path, path.basename(path.dirname(ground_truth_map_file_path))+'.pgm'))
    except IOError:
        print_error("Error while saving image {img}:".format(img=ground_truth_map_file_path))
        print_error(traceback.format_exc())
    except TypeError:
        print_error("Error while saving image {img}:".format(img=ground_truth_map_file_path))
        print_error(traceback.format_exc())


if __name__ == '__main__':
    environment_folders = sorted(glob.glob(path.expanduser("~/ds/performance_modelling/dataset/*")))
    dump_path = path.expanduser("~/tmp/gt_maps/")
    print_info("compute_ground_truth_from_grid_map {}%".format(0))
    for progress, environment_folder in enumerate(environment_folders):
        print_info("compute_ground_truth_from_grid_map {}% {}".format((progress + 1)*100//len(environment_folders), environment_folder))
        map_file_path = path.join(environment_folder, "data", "map.png")
        map_info_file_path = path.join(environment_folder, "data", "map_info.yaml")
        result_ground_truth_file_path = path.join(environment_folder, "data", "map_ground_truth.pgm")
        compute_ground_truth_from_grid_map(map_file_path, map_info_file_path, result_ground_truth_file_path, ground_truth_map_files_dump_path=dump_path)
