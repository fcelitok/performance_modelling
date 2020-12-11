#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import glob
import os
import pickle
import traceback
from os import path
import time
import networkx as nx
from PIL.ImageDraw import floodfill
from PIL import Image, ImageFilter, ImageChops
import numpy as np
import yaml
from performance_modelling_py.utils import print_info, backup_file_if_exists, print_error, print_fatal
from scipy import ndimage
from scipy.spatial import Delaunay


def trim(im, border_color):
    bg = Image.new(im.mode, im.size, border_color)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def cm_to_body_parts(*argv):
    inch = 2.54
    if isinstance(argv[0], tuple):
        return tuple(x_cm / inch for x_cm in argv[0])
    else:
        return tuple(x_cm / inch for x_cm in argv)


def circle_given_points(p1, p2, p3):
    """
    Center and radius of a circle given 3 points.
    """

    p12 = p1 - p2
    p23 = p2 - p3

    bc = (np.sum(p1**2) - np.sum(p2**2)) / 2
    cd = (np.sum(p2**2) - np.sum(p3**2)) / 2

    det = p12[0] * p23[1] - p23[0] * p12[1]

    if abs(det) < 1.0e-6:
        return (p1 + p2 + p3)/3, np.inf

    c = np.array((bc * p23[1] - cd * p12[1], cd * p12[0] - bc * p23[0])) / det
    r = np.sqrt(np.sum((c - p1)**2))

    return c, r


def black_white_to_ground_truth_map(input_map_path, map_info_path, trim_borders=False, occupied_threshold=150, blur_filter_radius=0, do_not_recompute=False, backup_if_exists=False, map_files_dump_path=None):
    with open(map_info_path) as map_info_file:
        map_info = yaml.safe_load(map_info_file)

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
        background = Image.new("RGB", input_image.size, GroundTruthMap.free_rgb)
        background.paste(input_image)
        input_image = background

    # apply a blur filter to reduce artifacts in images that have been saved to lossy formats (may cause the thickness of walls to change)
    if blur_filter_radius > 0:
        input_image = input_image.filter(ImageFilter.BoxBlur(blur_filter_radius))

    # saturate all colors below the threshold to black and all values above threshold to the unknown color value (they will be colored to free later)
    # this is needed because some images have been heavily compressed and contain all sort of colors
    threshold_image = input_image.point(lambda original_value: GroundTruthMap.occupied if original_value < occupied_threshold else GroundTruthMap.unknown)
    threshold_image.save(path.expanduser("~/tmp/threshold_image.pgm"))

    # trim borders not containing black pixels (some simulators ignore non-black borders while placing the pixels in the simulated map and computing its resolution, so they need to be ignored in the following calculations)
    if trim_borders:
        trimmed_image = trim(threshold_image, GroundTruthMap.unknown_rgb)
        trimmed_image.save(path.expanduser("~/tmp/trimmed_image.pgm"))
    else:
        trimmed_image = threshold_image

    map_offset_meters = np.array(map_info['origin'][0:2], dtype=float)

    map_rotation_offset = np.array(map_info['origin'][2:3], dtype=float)
    if map_rotation_offset != 0:
        print_error("convert_grid_map_to_gt_map: map rotation not supported")

    resolution = float(map_info['resolution'])  # meter/pixel
    map_frame_meters = -map_offset_meters

    if 'initial_pose' in map_info:
        initial_position_list = map_info['initial_pose']
    else:
        initial_position_list = [0.0, 0.0]
    initial_position_meters = np.array(initial_position_list, dtype=float)

    w, h = trimmed_image.size
    i_x, i_y = initial_position_pixels = list(map(int, np.array([0, h]) + np.array([1, -1]) * (map_frame_meters + initial_position_meters) / resolution))

    if i_x < 0 or i_x >= w or i_y < 0 or i_y >= h:
        print_fatal("initial_position out of map bounds")
        return

    pixels = trimmed_image.load()
    if pixels[i_x, i_y] != GroundTruthMap.unknown_rgb:
        print_fatal("initial position in a wall pixel")
        return

    # rename variable for clarity
    map_image = trimmed_image

    # convert to free the pixels accessible from the initial position
    floodfill(map_image, initial_position_pixels, GroundTruthMap.free_rgb, thresh=10)

    if backup_if_exists:
        backup_file_if_exists(map_image_path)

    try:
        map_image.save(map_image_path)
        if map_files_dump_path is not None:
            print(map_files_dump_path)
            map_files_dump_path = path.abspath(path.expanduser(map_files_dump_path))
            print(map_files_dump_path)
            if not path.exists(map_files_dump_path):
                os.makedirs(map_files_dump_path)
            dataset_name = path.basename(path.dirname(path.dirname(map_image_path)))
            map_image.save(path.join(map_files_dump_path, dataset_name + '.pgm'))
    except IOError:
        print_fatal("Error while saving image {img}:".format(img=map_image_path))
        print_error(traceback.format_exc())
    except TypeError:
        print_fatal("Error while saving image {img}:".format(img=map_image_path))
        print_error(traceback.format_exc())


class NotFoundException(Exception):
    pass


class GroundTruthMap:

    FREE = 0
    UNKNOWN = 1
    OCCUPIED = 2

    occupied = 0  # color value of occupied cells in the ground truth map image
    unknown = 205  # color value of unknown cells in the ground truth map image
    free = 255  # color value of free cells in the ground truth map image
    occupied_rgb = (occupied, occupied, occupied)
    unknown_rgb = (unknown, unknown, unknown)
    free_rgb = (free, free, free)

    def __init__(self, map_info_path):

        with open(map_info_path) as map_info_file:
            map_info = yaml.safe_load(map_info_file)

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

        if 'initial_pose' in map_info:
            self.initial_position = map_info['initial_pose']
        else:
            self.initial_position = [0.0, 0.0]

        self._complete_free_voronoi_graph_file_path = path.join(path.dirname(map_info_path), "complete_free_voronoi_graph_cache.pkl")

        self._occupancy_map = None
        self._complete_free_voronoi_graph = None

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

    def map_frame_to_image_coordinates(self, xy_meters):
        _, h = self.map_image.size
        return list(map(int, np.array([0, h]) + np.array([1, -1]) * (self.map_frame_meters + xy_meters) / self.resolution))

    def image_to_map_frame_coordinates(self, xy_pixels):
        _, h = self.map_image.size
        return self.resolution * (np.array([0, h]) + np.array([1, -1]) * xy_pixels) - self.map_frame_meters

    def image_y_up_to_map_frame_coordinates(self, p_pixels):
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

            sampled_position_x_meters, sampled_position_y_meters = self.image_to_map_frame_coordinates(np.array(x, y))
            sampled_orientation_radians = np.random.uniform(-np.pi, np.pi)

            if num_poses == 1:
                return sampled_position_x_meters, sampled_position_y_meters, sampled_orientation_radians
            else:
                sampled_poses.append((sampled_position_x_meters, sampled_position_y_meters, sampled_orientation_radians))
                if len(sampled_poses) == num_poses:
                    return sampled_poses

        raise NotFoundException("maximum number of attempts reached")

    def edge_bitmaps(self, occupancy_function):
        pixels = self.map_image.load()

        # occupied_bitmap contains 1 where pixels are occupied (black color value -> wall)
        # occupied_bitmap coordinates have origin in the image bottom-left with y-up rather than top-left with y-down,
        # so it is in between image coordinates and map frame coordinates
        w, h = self.map_image.size
        occupied_bitmap = np.zeros((w+1, h+1), dtype=int)
        for y in range(h):
            for x in range(w):
                occupied_bitmap[x, h-1-y] = occupancy_function(pixels[x, y])

        # span the image with kernels to find vertical walls, north -> +1, south -> -1
        vertical_edges = ndimage.convolve(occupied_bitmap, np.array([[1, -1]]), mode='constant', cval=0, origin=np.array([0, -1]))
        north_bitmap = vertical_edges == 1
        south_bitmap = vertical_edges == -1

        # span the image with kernels to find horizontal walls, west -> +1, east -> -1
        horizontal_edges = ndimage.convolve(occupied_bitmap, np.array([[1], [-1]]), mode='constant', cval=0, origin=np.array([-1, 0]))
        west_bitmap = horizontal_edges == 1
        east_bitmap = horizontal_edges == -1

        return occupied_bitmap, north_bitmap, south_bitmap, west_bitmap, east_bitmap

    def _compute_complete_free_voronoi_graph(self):
        occupied_bitmap, north_bitmap, south_bitmap, west_bitmap, east_bitmap = self.edge_bitmaps(lambda pixel: pixel != self.free_rgb)

        wall_points_set = set()
        w, h = north_bitmap.shape
        for x in range(w):
            for y in range(h):
                if north_bitmap[x, y] or south_bitmap[x, y]:
                    wall_points_set.add((x, y))
                    wall_points_set.add((x + 1, y))
                if west_bitmap[x, y] or east_bitmap[x, y]:
                    wall_points_set.add((x, y))
                    wall_points_set.add((x, y + 1))

        wall_points = np.array(tuple(wall_points_set))
        delaunay_graph = Delaunay(wall_points)

        vertices_meters_list = list()
        vertices_pixels_list = list()
        radii_meters_list = list()
        for vertex_indices in delaunay_graph.simplices:
            p1_p, p2_p, p3_p = wall_points[vertex_indices, :]
            center_p, _ = circle_given_points(p1_p, p2_p, p3_p)
            vertices_pixels_list.append(center_p)

            p1_m, p2_m, p3_m = self.image_y_up_to_map_frame_coordinates(wall_points[vertex_indices, :])
            center_m, radius_m = circle_given_points(p1_m, p2_m, p3_m)
            vertices_meters_list.append(center_m)
            radii_meters_list.append(radius_m)

        voronoi_vertices_meters = np.array(vertices_meters_list)
        voronoi_vertices_int_pixels = np.array(vertices_pixels_list, dtype=int)
        voronoi_vertices_occupancy_bitmap = occupied_bitmap[voronoi_vertices_int_pixels[:, 0], voronoi_vertices_int_pixels[:, 1]]

        free_indices_set = set(list(np.where(1 - voronoi_vertices_occupancy_bitmap)[0]))

        complete_free_voronoi_graph = nx.Graph()
        for triangle_index, (n1, n2, n3) in enumerate(delaunay_graph.neighbors):
            if triangle_index in free_indices_set:
                complete_free_voronoi_graph.add_node(
                    triangle_index,
                    vertex=voronoi_vertices_meters[triangle_index, :],
                    radius=radii_meters_list[triangle_index])

                if n1 < triangle_index and n1 != -1 and n1 in free_indices_set:
                    complete_free_voronoi_graph.add_edge(triangle_index, int(n1))
                if n2 < triangle_index and n2 != -1 and n2 in free_indices_set:
                    complete_free_voronoi_graph.add_edge(triangle_index, int(n2))
                if n3 < triangle_index and n3 != -1 and n3 in free_indices_set:
                    complete_free_voronoi_graph.add_edge(triangle_index, int(n3))

        for n1, n2 in list(complete_free_voronoi_graph.edges):
            p1 = complete_free_voronoi_graph.nodes[n1]['vertex']
            p2 = complete_free_voronoi_graph.nodes[n2]['vertex']
            complete_free_voronoi_graph.edges[n1, n2]['voronoi_path_distance'] = np.sqrt(np.sum((p2 - p1) ** 2))

        return complete_free_voronoi_graph

    @property
    def voronoi_graph(self):
        # if the graph has not been already computed in this object, we need to load it from file or compute it
        if self._complete_free_voronoi_graph is None:

            # if the cache file exists, we load the graph from file
            if path.exists(self._complete_free_voronoi_graph_file_path):
                with open(self._complete_free_voronoi_graph_file_path, 'rb') as complete_free_voronoi_graph_file:
                    self._complete_free_voronoi_graph = pickle.load(complete_free_voronoi_graph_file)
            # otherwise we compute it and save it to file
            else:
                self._complete_free_voronoi_graph = self._compute_complete_free_voronoi_graph()
                with open(self._complete_free_voronoi_graph_file_path, 'wb') as complete_free_voronoi_graph_file:
                    pickle.dump(self._complete_free_voronoi_graph, complete_free_voronoi_graph_file, protocol=2)

        return self._complete_free_voronoi_graph

    def reduced_voronoi_graph(self, minimum_radius):

        min_radius_voronoi_graph = self.voronoi_graph.subgraph(filter(
            lambda n: self.voronoi_graph.nodes[n]['radius'] >= minimum_radius,
            self.voronoi_graph.nodes
        ))

        chain_nodes = list(filter(
            lambda n: len(list(min_radius_voronoi_graph.neighbors(n))) == 2,
            min_radius_voronoi_graph.nodes
        ))

        reduced_voronoi_graph = min_radius_voronoi_graph.copy()
        for n2 in chain_nodes:
            n1, n3 = reduced_voronoi_graph.neighbors(n2)
            w1 = reduced_voronoi_graph.edges[n1, n2]['voronoi_path_distance']
            w2 = reduced_voronoi_graph.edges[n2, n3]['voronoi_path_distance']
            reduced_voronoi_graph.remove_node(n2)
            reduced_voronoi_graph.add_edge(n1, n3, voronoi_path_distance=w1+w2)

        return reduced_voronoi_graph

    def deleaved_reduced_voronoi_graph(self, minimum_radius):

        deleaved_reduced_voronoi_graph = self.reduced_voronoi_graph(minimum_radius).copy()
        leaf_nodes = list(filter(
            lambda n: len(list(deleaved_reduced_voronoi_graph.neighbors(n))) == 1,
            deleaved_reduced_voronoi_graph.nodes
        ))
        deleaved_reduced_voronoi_graph.remove_nodes_from(leaf_nodes)

        return deleaved_reduced_voronoi_graph

    def save_voronoi_plot(self, plot_file_path, graph=None, do_not_recompute=False, timeout=120, max_nodes=2000, min_radius=None):
        if path.exists(plot_file_path) and do_not_recompute:
            print_info("do_not_recompute: will not recompute the voronoi plot {}".format(plot_file_path))
            return

        if not path.exists(path.dirname(plot_file_path)):
            os.makedirs(path.dirname(plot_file_path))

        if min_radius is None:
            min_radius = 4 * self.resolution

        if graph is None:
            graph = self.voronoi_graph.subgraph(filter(
                lambda n: self.voronoi_graph.nodes[n]['radius'] >= min_radius,
                self.voronoi_graph.nodes
            ))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig_size = cm_to_body_parts(5 * self.map_size_meters)
        fig.set_size_inches(*fig_size)

        start_time = time.time()
        print("plotting. nodes: {}/{}".format(min(max_nodes, graph.number_of_nodes()), graph.number_of_nodes()))
        print("          edges: {}/{}".format(min(max_nodes, graph.number_of_nodes()), graph.number_of_edges()))
        print("          fig size: {} cm".format(fig_size))

        num_nodes = 0
        nth = max(1, graph.number_of_nodes() // max_nodes)
        for node_index, node_data in graph.nodes.data():
            num_nodes += 1
            if num_nodes % nth:
                continue

            x_1, y_1 = node_data['vertex']
            radius_1 = node_data['radius']

            if radius_1 < min_radius:
                continue

            # plot leaf vertices
            if len(list(graph.neighbors(node_index))) == 1:
                ax.scatter(x_1, y_1, color='red', s=4.0, marker='o')

            # plot chain vertices
            if len(list(graph.neighbors(node_index))) == 3:
                ax.scatter(x_1, y_1, color='blue', s=4.0, marker='o')

            # plot segments
            for neighbor_index in graph.neighbors(node_index):
                if neighbor_index < node_index:
                    radius_2 = graph.nodes[neighbor_index]['radius']
                    if radius_2 > min_radius:
                        x_2, y_2 = graph.nodes[neighbor_index]['vertex']
                        ax.plot((x_1, x_2), (y_1, y_2), color='black', linewidth=1.0)

            # plot circles
            ax.add_artist(plt.Circle(node_data['vertex'], radius_1, color='grey', fill=False, linewidth=0.2))

            if time.time() - start_time > timeout:
                print("timeout")
                break

        # plot vertical and horizontal wall points
        _, north_bitmap, south_bitmap, west_bitmap, east_bitmap = self.edge_bitmaps(lambda pixel: pixel != self.free_rgb)
        h_wall_points_pixels_set = set()
        v_wall_points_pixels_set = set()

        w, h = north_bitmap.shape
        for x in range(w):
            for y in range(h):
                if north_bitmap[x, y] or south_bitmap[x, y]:
                    h_wall_points_pixels_set.add((x, y))
                    h_wall_points_pixels_set.add((x + 1, y))
                if west_bitmap[x, y] or east_bitmap[x, y]:
                    v_wall_points_pixels_set.add((x, y))
                    v_wall_points_pixels_set.add((x, y + 1))

        h_wall_points_meters = self.image_y_up_to_map_frame_coordinates(np.array(list(h_wall_points_pixels_set)))
        v_wall_points_meters = self.image_y_up_to_map_frame_coordinates(np.array(list(v_wall_points_pixels_set)))

        ax.scatter(h_wall_points_meters[:, 0], h_wall_points_meters[:, 1], s=15.0, marker='_')
        ax.scatter(v_wall_points_meters[:, 0], v_wall_points_meters[:, 1], s=15.0, marker='|')

        print("saving plot:", plot_file_path)
        fig.savefig(plot_file_path)
        plt.close(fig)

    def save_voronoi_plot_and_trajectory(self, plot_file_path, segments, graph=None, do_not_recompute=False, timeout=120, max_nodes=2000, min_radius=None):
        plot_file_path = path.expanduser(plot_file_path)
        if path.exists(plot_file_path) and do_not_recompute:
            print_info("do_not_recompute: will not recompute the voronoi plot {}".format(plot_file_path))
            return

        if not path.exists(path.dirname(plot_file_path)):
            os.makedirs(path.dirname(plot_file_path))

        if min_radius is None:
            min_radius = 4 * self.resolution

        if graph is None:
            graph = self.voronoi_graph.subgraph(filter(
                lambda n: self.voronoi_graph.nodes[n]['radius'] >= min_radius,
                self.voronoi_graph.nodes
            ))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig.set_size_inches(*cm_to_body_parts(10 * self.map_size_meters))

        start_time = time.time()
        print("plotting. nodes: {}/{}".format(min(max_nodes, graph.number_of_nodes()), graph.number_of_nodes()))

        for (x_1, y_1), (x_2, y_2) in segments:
            ax.plot((x_1, x_2), (y_1, y_2), color='blue', linewidth=1.5)

        num_nodes = 0
        nth = max(1, graph.number_of_nodes() // max_nodes)
        for node_index, node_data in graph.nodes.data():
            num_nodes += 1
            if num_nodes % nth:
                continue

            x_1, y_1 = node_data['vertex']
            radius_1 = node_data['radius']

            if radius_1 < min_radius:
                continue

            # plot leaf vertices
            if len(list(graph.neighbors(node_index))) == 1:
                ax.scatter(x_1, y_1, color='red', s=4.0, marker='o')

            # plot chain vertices
            if len(list(graph.neighbors(node_index))) == 3:
                ax.scatter(x_1, y_1, color='blue', s=4.0, marker='o')

            # plot other vertices
            if len(list(graph.neighbors(node_index))) == 2:
                ax.scatter(x_1, y_1, color='grey', s=2.0, marker='o')

            # plot segments
            for neighbor_index in graph.neighbors(node_index):
                if neighbor_index < node_index:
                    radius_2 = graph.nodes[neighbor_index]['radius']
                    if radius_2 > min_radius:
                        x_2, y_2 = graph.nodes[neighbor_index]['vertex']
                        ax.plot((x_1, x_2), (y_1, y_2), color='black', linewidth=1.0)

            # plot circles
            ax.add_artist(plt.Circle(node_data['vertex'], radius_1, color='grey', fill=False, linewidth=0.2))

            if time.time() - start_time > timeout:
                print("timeout")
                break

        # plot vertical and horizontal wall points
        _, north_bitmap, south_bitmap, west_bitmap, east_bitmap = self.edge_bitmaps(lambda pixel: pixel != self.free_rgb)
        h_wall_points_pixels_set = set()
        v_wall_points_pixels_set = set()

        w, h = north_bitmap.shape
        for x in range(w):
            for y in range(h):
                if north_bitmap[x, y] or south_bitmap[x, y]:
                    h_wall_points_pixels_set.add((x, y))
                    h_wall_points_pixels_set.add((x + 1, y))
                if west_bitmap[x, y] or east_bitmap[x, y]:
                    v_wall_points_pixels_set.add((x, y))
                    v_wall_points_pixels_set.add((x, y + 1))

        h_wall_points_meters = self.image_y_up_to_map_frame_coordinates(np.array(list(h_wall_points_pixels_set)))
        v_wall_points_meters = self.image_y_up_to_map_frame_coordinates(np.array(list(v_wall_points_pixels_set)))

        ax.scatter(h_wall_points_meters[:, 0], h_wall_points_meters[:, 1], s=15.0, marker='_')
        ax.scatter(v_wall_points_meters[:, 0], v_wall_points_meters[:, 1], s=15.0, marker='|')

        fig.savefig(plot_file_path)
        plt.close(fig)


if __name__ == '__main__':
    default_environment_folders = "~/ds/performance_modelling/test_datasets/dataset/*"
    default_dump_path = "~/tmp/gt_maps/"

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Computes ground truth map, mesh and Voronoi plots from source image.')
    parser.add_argument('-e', dest='environment_folders',
                        help='Folder containing the datasets. Example: {}'.format(default_environment_folders),
                        type=str,
                        required=True)

    parser.add_argument('-d', dest='dump_path',
                        help='Folder in which to dump a copy of the generated map images. Defaults to {}'.format(default_dump_path),
                        type=str,
                        default=default_dump_path,
                        required=False)

    parser.add_argument('-r', dest='recompute_data',
                        help='If set, the data is re-computed.',
                        action='store_true',
                        default=False,
                        required=False)

    parser.add_argument('-p', dest='save_visualization_plots',
                        help='If set, the Voronoi visualization plots are computed and saved.',
                        action='store_true',
                        default=False,
                        required=False)

    args = parser.parse_args()

    environment_folders = sorted(filter(path.isdir, glob.glob(path.expanduser(args.environment_folders))))
    dump_path = args.dump_path

    print_info("computing environment data {}%".format(0))
    recompute_data = args.recompute_data
    save_visualization_plots = args.save_visualization_plots
    recompute_plots = True

    for progress, environment_folder in enumerate(environment_folders):
        print_info("computing environment data {}% {}".format(progress * 100 // len(environment_folders), environment_folder))

        map_info_file_path = path.join(environment_folder, "data", "map.yaml")

        # compute GroundTruthMap data from source image
        source_map_file_path = None
        source_pgm_map_file_path = path.join(environment_folder, "data", "source_map.pgm")
        source_png_map_file_path = path.join(environment_folder, "data", "source_map.png")
        if path.exists(source_pgm_map_file_path):
            source_map_file_path = source_pgm_map_file_path
        elif path.exists(source_png_map_file_path):
            source_map_file_path = source_png_map_file_path
        else:
            print_error("source_map file not found")

        if source_map_file_path is not None:
            black_white_to_ground_truth_map(source_map_file_path, map_info_file_path, do_not_recompute=not recompute_data, map_files_dump_path=dump_path)

        if save_visualization_plots:
            # compute voronoi plot
            robot_radius_plot = 2.5 * 0.2
            voronoi_plot_file_path = path.join(environment_folder, "data", "visualization", "voronoi.svg")
            reduced_voronoi_plot_file_path = path.join(environment_folder, "data", "visualization", "reduced_voronoi.svg")
            deleaved_reduced_voronoi_plot_file_path = path.join(environment_folder, "data", "visualization", "deleaved_reduced_voronoi.svg")
            m = GroundTruthMap(map_info_file_path)
            m.save_voronoi_plot(voronoi_plot_file_path, min_radius=robot_radius_plot, do_not_recompute=not recompute_plots)
            m.save_voronoi_plot(reduced_voronoi_plot_file_path, graph=m.reduced_voronoi_graph(minimum_radius=robot_radius_plot), min_radius=robot_radius_plot, do_not_recompute=not recompute_plots)
            m.save_voronoi_plot(deleaved_reduced_voronoi_plot_file_path, graph=m.deleaved_reduced_voronoi_graph(robot_radius_plot), min_radius=robot_radius_plot, do_not_recompute=not recompute_plots)

        # compute mesh
        from performance_modelling_py.environment.mesh_utils import gridmap_to_mesh
        result_mesh_file_path = path.join(environment_folder, "data", "meshes", "extruded_map.dae")
        gridmap_to_mesh(map_info_file_path, result_mesh_file_path, do_not_recompute=not recompute_data)

        print_info("computing environment data {}% done".format((progress + 1) * 100 // len(environment_folders)))
