#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import yaml
from PIL import Image
from performance_modelling_py.utils import print_info


def color_diff(a, b):
    return np.sum(np.array(b) - np.array(a)) / len(a)


def color_abs_diff(a, b):
    return np.abs(color_diff(a, b))


def rgb_less_than(a, b):
    return a[0] < b[0] and a[1] < b[1] and a[2] < b[2]


class NotFoundException(Exception):
    pass


class GroundTruthMap:

    FREE = 0
    UNKNOWN = 1
    OCCUPIED = 2

    def __init__(self, map_image_path, map_info_path):
        print_info(map_image_path)
        self.map_image = Image.open(map_image_path)
        self.width, self.height = self.map_image.size
        with open(map_info_path) as map_info_file:
            self.info_dict = yaml.load(map_info_file)

        self.unknown = 205  # color value of unknown cells in the ground truth map image
        self.map_size_meters = np.array([float(self.info_dict['map']['size']['x']), float(self.info_dict['map']['size']['y'])])
        self.map_size_pixels = np.array(self.map_image.size)
        self.map_offset_meters = np.array([float(self.info_dict['map']['pose']['x']), float(self.info_dict['map']['pose']['y'])])
        self.resolution = self.map_size_meters / self.map_size_pixels * np.array([1, 1])  # meter/pixel, on both axis

        self._occupancy_map = None

    @property
    def occupancy_map(self):
        if self._occupancy_map is None:
            # convert colors to occupancy values
            pixels = self.map_image.load()
            self._occupancy_map = np.empty(self.map_image.size, dtype=int)
            for i in range(self.width):
                for j in range(self.height):
                    if pixels[i, j][0] < self.unknown:
                        self._occupancy_map[i, j] = self.OCCUPIED
                    elif pixels[i, j][0] == self.unknown:
                        self._occupancy_map[i, j] = self.UNKNOWN
                    else:
                        self._occupancy_map[i, j] = self.FREE
        return self._occupancy_map

    def pixels_to_world_coordinates(self, x_pixels, y_pixels):
        p_pixels = np.array([x_pixels, self.height - y_pixels])
        return self.resolution * p_pixels - 0.5 * self.map_size_meters + self.map_offset_meters

    def sample_robot_pose_from_free_cells(self, robot_radius, num_poses=1, max_attempts=100):
        if num_poses <= 0:
            return

        free_cell_bitmap = self.occupancy_map == self.FREE
        i_mask = np.array(np.where(free_cell_bitmap))  # array of indices of the free cells
        num_choices = i_mask.shape[1]  # number of free cells

        if num_choices == 0:
            raise NotFoundException("GroundTruthMap does not contain free cells")

        sampled_poses = list()
        for _ in range(max_attempts*num_poses):
            # choose a random index where to position the robot
            x, y = i_mask[:, np.random.choice(range(num_choices))]

            # check the robot (approximated) footprint is also contained in the free cells
            margin = 2  # additional margin to guarantee a distance of at least one pixel between the actual footprint and non-free cells
            footprint_w, footprint_h = (2*robot_radius / self.resolution) + margin
            footprint_x_min, footprint_x_max = int(np.floor(x - footprint_w/2)), int(np.ceil(x + footprint_w/2)) + 1  # + 1 because the range needs to be inclusive
            footprint_y_min, footprint_y_max = int(np.floor(y - footprint_h/2)), int(np.ceil(y + footprint_h/2)) + 1  # + 1 because the range needs to be inclusive

            # footprint out of map bounds
            if footprint_x_min < 0 or footprint_x_max > self.width or footprint_y_min < 0 or footprint_y_max > self.height:
                continue

            # sub-array of the occupancy map corresponding to the robot footprint
            footprint_free_cell_bitmap = free_cell_bitmap[footprint_x_min:footprint_x_max, footprint_y_min:footprint_y_max]

            # footprint intersects non-free cells
            if not np.all(footprint_free_cell_bitmap):
                continue

            sampled_position_x_meters, sampled_position_y_meters = self.pixels_to_world_coordinates(x, y)
            sampled_orientation_radians = np.random.uniform(-np.pi, np.pi)

            if num_poses == 1:
                return sampled_position_x_meters, sampled_position_y_meters, sampled_orientation_radians
            else:
                sampled_poses.append((sampled_position_x_meters, sampled_position_y_meters, sampled_orientation_radians))
                if len(sampled_poses) == num_poses:
                    return sampled_poses

        raise NotFoundException("maximum number of attempts reached")
