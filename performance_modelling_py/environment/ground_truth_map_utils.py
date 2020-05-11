#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
import traceback
from os import path
from PIL.ImageDraw import floodfill
from PIL import Image, ImageFilter, ImageChops

import numpy as np
import yaml

from performance_modelling_py.utils import print_info, backup_file_if_exists, print_error, print_fatal


def trim(im, border_color):
    bg = Image.new(im.mode, im.size, border_color)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def black_white_to_ground_truth_map(input_map_path, map_info_path, unknown=205, occupied_threshold=150, blur_filter_radius=0, do_not_recompute=False, backup_if_exists=False, map_files_dump_path=None):
    unknown_rgb = (unknown, unknown, unknown)  # color value of unknown cells in the ground truth map

    with open(map_info_path) as map_info_file:
        map_info = yaml.load(map_info_file)

    if path.isabs(map_info['image']):
        map_image_path = map_info['image']
    else:
        map_image_path = path.join(path.dirname(map_info_path), map_info['image'])

    if path.exists(map_image_path) and do_not_recompute:
        print_info("do_not_recompute: will not recompute image {}".format(map_image_path))
        return

    input_image = Image.open(input_map_path)

    if input_image.mode != 'RGB':
        # remove alpha channel by pasting on white background
        background = Image.new("RGB", input_image.size, (254, 254, 254))
        background.paste(input_image)
        input_image = background

    # apply a blur filter to reduce artifacts in images that have been saved to lossy formats (may cause the thickness of walls to change)
    if blur_filter_radius > 0:
        input_image = input_image.filter(ImageFilter.BoxBlur(blur_filter_radius))

    # saturate all colors below the threshold to black and all values above threshold to the unknown color value (they will be colored to free later)
    # this is needed because some images have been heavily compressed and contain all sort of colors
    threshold_image = input_image.point(lambda original_value: 0 if original_value < occupied_threshold else unknown)
    threshold_image.save(path.expanduser("~/tmp/threshold_image.pgm"))

    # trim borders not containing black pixels (some simulators ignore non-black borders while placing the pixels in the simulated map and computing its resolution, so they need to be ignored in the following calculations)
    trimmed_image = trim(threshold_image, unknown_rgb)
    trimmed_image.save(path.expanduser("~/tmp/trimmed_image.pgm"))

    map_offset_meters = np.array(map_info['origin'][0:2], dtype=float)

    map_rotation_offset = np.array(map_info['origin'][2:3], dtype=float)
    if map_rotation_offset != 0:
        print_error("convert_grid_map_to_gt_map: map rotation not supported")

    resolution = float(map_info['resolution'])  # meter/pixel
    map_frame_meters = -map_offset_meters
    initial_position_meters = np.array(map_info['initial_pose'], dtype=float)

    w, h = trimmed_image.size
    i_x, i_y = initial_position_pixels = list(map(int, np.array([0, h]) + np.array([1, -1]) * (map_frame_meters + initial_position_meters) / resolution))

    if i_x < 0 or i_x >= w or i_y < 0 or i_y >= h:
        print_fatal("initial_position out of map bounds")
        return

    pixels = trimmed_image.load()
    if pixels[i_x, i_y] != unknown_rgb:
        print_fatal("initial position in a wall pixel")
        return

    # rename variable for clarity
    map_image = trimmed_image

    # convert to free the pixels accessible from the initial position
    floodfill(map_image, initial_position_pixels, (254, 254, 254), thresh=10)

    if backup_if_exists:
        backup_file_if_exists(map_image_path)

    try:
        map_image.save(map_image_path)
        if map_files_dump_path is not None:
            if not path.exists(map_files_dump_path):
                os.makedirs(path.basename(map_files_dump_path))
            dataset_name = path.basename(path.dirname(path.dirname(map_image_path)))
            map_image.save(path.join(map_files_dump_path, dataset_name + '.pgm'))
    except IOError:
        print_fatal("Error while saving image {img}:".format(img=map_image_path))
        print_error(traceback.format_exc())
    except TypeError:
        print_fatal("Error while saving image {img}:".format(img=map_image_path))
        print_error(traceback.format_exc())


if __name__ == '__main__':
    environment_folders = sorted(glob.glob(path.expanduser("~/ds/performance_modelling/dataset/*")))
    dump_path = path.expanduser("~/tmp/gt_maps/")
    print_info("compute_ground_truth_from_grid_map {}%".format(0))
    for progress, environment_folder in enumerate(environment_folders):
        print_info("compute_ground_truth_from_grid_map {}% {}".format((progress + 1)*100//len(environment_folders), environment_folder))
        source_map_file_path = path.join(environment_folder, "data", "source_map.png")
        map_info_file_path = path.join(environment_folder, "data", "map.yaml")
        black_white_to_ground_truth_map(source_map_file_path, map_info_file_path, map_files_dump_path=dump_path)
