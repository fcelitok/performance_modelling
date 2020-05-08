#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import traceback
from os import path
from PIL.ImageDraw import floodfill
from PIL import Image

import numpy as np
import yaml

from performance_modelling_py.utils import print_info, backup_file_if_exists, print_error, print_fatal


def color_diff(a, b):
    return np.sum(np.array(b) - np.array(a)) / len(a)


def color_abs_diff(a, b):
    return np.abs(color_diff(a, b))


def compute_ground_truth_from_grid_map(grid_map_file_path, grid_map_info_file_path, ground_truth_map_file_path, do_not_recompute=False, backup_if_exists=False, ground_truth_map_files_dump_path=None):

    if path.exists(ground_truth_map_file_path):
        print_info("file already exists: {}".format(ground_truth_map_file_path))
        if do_not_recompute:
            print_info("do_not_recompute: will not recompute the output image")
            return

    gt = Image.open(grid_map_file_path)
    print_info("opened image, mode: {mode}, image: {im}".format(mode=gt.mode, im=grid_map_file_path))

    if gt.mode != 'RGB':
        print('image mode is {mode} ({size}×{ch_num}), converting to RGB'.format(mode=gt.mode, size=gt.size, ch_num=len(gt.split())))
        # remove alpha channel by pasting on white background
        background = Image.new("RGB", gt.size, (254, 254, 254))
        background.paste(gt)
        gt = background

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

    print("crop box: left_border: {}, top_border: {}, right_border: {}, bottom_border: {}".format(left_border, top_border, right_border, bottom_border))
    gt = gt.crop(box=(left_border, top_border, right_border, bottom_border))
    pixels = gt.load()
    print('cropped image: {mode} ({size}×{ch_num})'.format(mode=gt.mode, size=gt.size, ch_num=len(gt.split())))

    # convert all free pixels to unknown pixels
    for i in range(gt.size[0]):
        for j in range(gt.size[1]):
            if color_abs_diff(pixels[i, j], (0, 0, 0)) < 150:  # black -> wall. color_abs_diff must be less than 205 - 0 (difference between occupied black and unknown grey)
                pixels[i, j] = (0, 0, 0)
            else:  # not black -> unknown space
                pixels[i, j] = (205, 205, 205)

    with open(grid_map_info_file_path, 'r') as info_file:
        info_yaml = yaml.load(info_file)

    if info_yaml['map']['pose']['x'] != 0 or info_yaml['map']['pose']['y'] != 0 or info_yaml['map']['pose']['z'] != 0 or info_yaml['map']['pose']['theta'] != 0:
        print_error("convert_grid_map_to_gt_map: map not in origin")

    initial_position_meters = np.array([float(info_yaml['robot']['pose']['x']), float(info_yaml['robot']['pose']['y'])])
    print("initial position (meters):", initial_position_meters)

    map_size_meters = np.array([float(info_yaml['map']['size']['x']), float(info_yaml['map']['size']['y'])])
    print("map_size (meters):", map_size_meters)
    map_size_pixels = np.array(gt.size)
    print("map_size (pixels):", map_size_pixels)
    resolution = map_size_meters / map_size_pixels * np.array([1, -1])  # meter/pixel, on both axis, except y axis is inverted in image
    print("resolution:", resolution)

    map_center_pixels = map_size_pixels / 2
    print("map center (pixels):", map_center_pixels)
    p_x, p_y = initial_position_pixels = map(int, map_center_pixels + initial_position_meters / resolution)
    print("initial position (pixels):", initial_position_pixels)

    if pixels[p_x, p_y] != (205, 205, 205):
        print_fatal("initial position in a wall pixel")
        return

    # convert to free the pixels accessible from the initial pose
    floodfill(gt, initial_position_pixels, (254, 254, 254), thresh=10)

    if backup_if_exists:
        backup_file_if_exists(ground_truth_map_file_path)

    try:
        print_info("writing to {}".format(ground_truth_map_file_path))
        gt.save(ground_truth_map_file_path)
        if ground_truth_map_files_dump_path is not None:
            os.makedirs(path.basename(ground_truth_map_files_dump_path))
            gt.save(path.join(ground_truth_map_files_dump_path, path.basename(path.dirname(ground_truth_map_file_path))+'.pgm'))
    except IOError:
        print_error("Error while saving image {img}:".format(img=ground_truth_map_file_path))
        print_error(traceback.format_exc())
    except TypeError:
        print_error("Error while saving image {img}:".format(img=ground_truth_map_file_path))
        print_error(traceback.format_exc())


if __name__ == '__main__':
    environment_folder = path.expanduser("~/ds/performance_modelling_test_datasets/test/")
    map_file_path = path.join(environment_folder, "map.png")
    map_info_file_path = path.join(environment_folder, "grid_world_info.yaml")
    result_ground_truth_map_file_path = path.join(environment_folder, 'map_ground_truth.pgm')
    common_ground_truth_map_folder_path = path.expanduser("~/tmp/ground_truth_maps")

    compute_ground_truth_from_grid_map(map_file_path, map_info_file_path, result_ground_truth_map_file_path)
