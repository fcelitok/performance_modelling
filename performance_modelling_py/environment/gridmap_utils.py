#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

from os import path

import numpy as np
import yaml
from PIL import Image
from performance_modelling_py.utils import print_error


class NotFoundException(Exception):
    pass


class GroundTruthMap:

    FREE = 0
    UNKNOWN = 1
    OCCUPIED = 2
    unknown = 205  # color value of unknown cells in the ground truth map image

    def __init__(self, map_info_path):

        with open(map_info_path) as map_info_file:
            map_info = yaml.load(map_info_file)

        if path.isabs(map_info['image']):
            map_image_path = map_info['image']
        else:
            map_image_path = path.join(path.dirname(map_info_path), map_info['image'])

        self.map_image = Image.open(map_image_path)
        self.width, self.height = self.map_image.size
        self.map_offset_meters = np.array(map_info['origin'][0:2], dtype=float)

        map_rotation_offset = np.array(map_info['origin'][2:3], dtype=float)
        if map_rotation_offset != 0:
            print_error("convert_grid_map_to_gt_map: map rotation not supported")

        self.resolution = float(map_info['resolution'])  # meter/pixel
        self.map_size_pixels = np.array(self.map_image.size)
        self.map_size_meters = self.map_size_pixels * self.resolution
        self.map_frame_meters = -self.map_offset_meters

        self._occupancy_map = None

    @property
    def occupancy_map(self):
        if self._occupancy_map is None:
            # convert colors to occupancy values
            pixels = self.map_image.load()
            self._occupancy_map = np.empty(self.map_image.size, dtype=int)
            for x in range(self.width):
                for y in range(self.height):
                    if pixels[x, y][0] < self.unknown:
                        self._occupancy_map[x, y] = self.OCCUPIED
                    elif pixels[x, y][0] == self.unknown:
                        self._occupancy_map[x, y] = self.UNKNOWN
                    else:
                        self._occupancy_map[x, y] = self.FREE
        return self._occupancy_map

    def map_frame_to_image_coordinates(self, x_meters, y_meters):
        _, h = self.map_image.size
        p_meters = np.array([x_meters, y_meters])
        return list(map(int, np.array([0, h]) + np.array([1, -1]) * (self.map_frame_meters + p_meters) / self.resolution))

    def image_to_map_frame_coordinates(self, x_pixels, y_pixels):
        _, h = self.map_image.size
        p_pixels = np.array([x_pixels, y_pixels])
        return self.resolution * (np.array([0, h]) + np.array([1, -1]) * p_pixels) - self.map_frame_meters

    def image_y_up_to_map_frame_coordinates(self, x_pixels, y_pixels):
        p_pixels = np.array([x_pixels, y_pixels])
        return self.resolution * p_pixels - self.map_frame_meters

    def sample_robot_pose_from_free_cells(self, robot_radius, num_poses=1, max_attempts=100):
        if num_poses <= 0:
            return

        free_cell_bitmap = self.occupancy_map == self.FREE
        free_cell_indices = np.array(np.where(free_cell_bitmap))  # array of indices of the free cells
        num_choices = free_cell_indices.shape[1]  # number of free cells

        if num_choices == 0:
            raise NotFoundException("GroundTruthMap does not contain free cells")

        sampled_poses = list()
        for _ in range(max_attempts*num_poses):
            # choose a random index where to position the robot
            x, y = free_cell_indices[:, np.random.choice(range(num_choices))] + np.array([0.5, 0.5])  # add [0.5, 0.5] to position the image coordinates in the center of the chosen pixel

            # check the robot (approximated) footprint is also contained in the free cells
            margin = 2  # additional margin to guarantee a distance of at least one pixel between the actual footprint and non-free cells
            footprint_width = (2*robot_radius / self.resolution) + margin
            footprint_x_min, footprint_x_max = int(np.floor(x - footprint_width/2)), int(np.ceil(x + footprint_width/2)) + 1  # + 1 because the range needs to be inclusive
            footprint_y_min, footprint_y_max = int(np.floor(y - footprint_width/2)), int(np.ceil(y + footprint_width/2)) + 1  # + 1 because the range needs to be inclusive

            # footprint out of map bounds
            if footprint_x_min < 0 or footprint_x_max > self.width or footprint_y_min < 0 or footprint_y_max > self.height:
                continue

            # sub-array of the occupancy map corresponding to the robot footprint
            footprint_free_cell_bitmap = free_cell_bitmap[footprint_x_min:footprint_x_max, footprint_y_min:footprint_y_max]

            # footprint intersects non-free cells
            if not np.all(footprint_free_cell_bitmap):
                continue

            sampled_position_x_meters, sampled_position_y_meters = self.image_to_map_frame_coordinates(x, y)
            sampled_orientation_radians = np.random.uniform(-np.pi, np.pi)

            if num_poses == 1:
                return sampled_position_x_meters, sampled_position_y_meters, sampled_orientation_radians
            else:
                sampled_poses.append((sampled_position_x_meters, sampled_position_y_meters, sampled_orientation_radians))
                if len(sampled_poses) == num_poses:
                    return sampled_poses

        raise NotFoundException("maximum number of attempts reached")
