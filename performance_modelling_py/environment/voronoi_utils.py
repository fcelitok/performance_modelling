#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
from os import path
import numpy as np
from performance_modelling_py.environment.ground_truth_map_utils import GroundTruthMap
from performance_modelling_py.utils import print_info
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import networkx as nx


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


def gridmap_to_voronoi_graph(grid_map_info_file_path, voronoi_plot_file_path, do_not_recompute=False):

    m = GroundTruthMap(grid_map_info_file_path)

    occupied_bitmap, north_bitmap, south_bitmap, west_bitmap, east_bitmap = m.edge_bitmaps(lambda pixel: pixel != m.free_rgb)

    v_wall_points = set()
    h_wall_points = set()

    w, h = north_bitmap.shape
    for x in range(w):
        for y in range(h):
            if north_bitmap[x, y] or south_bitmap[x, y]:
                v_wall_points.add((x, y))
                v_wall_points.add((x+1, y))
            if west_bitmap[x, y] or east_bitmap[x, y]:
                h_wall_points.add((x, y))
                h_wall_points.add((x, y+1))

    corners = v_wall_points.union(h_wall_points)
    points = np.array(tuple(corners))
    delaunay_graph = Delaunay(points)

    centers_list = list()
    radii_list = list()
    for vertex_indices in delaunay_graph.simplices:
        p1, p2, p3 = points[vertex_indices, :]
        center, radius = circle_given_points(p1, p2, p3)
        centers_list.append(center)
        radii_list.append(radius)

    voronoi_vertices = np.array(centers_list)
    voronoi_radii = np.array(radii_list)

    voronoi_vertices_int_pixels = np.array(voronoi_vertices, dtype=int)
    voronoi_vertices_occupancy_bitmap = occupied_bitmap[voronoi_vertices_int_pixels[:, 0], voronoi_vertices_int_pixels[:, 1]]

    free_indices_set = set(list(np.where(1 - voronoi_vertices_occupancy_bitmap)[0]))

    segments_graph = nx.Graph()

    for triangle_index, (n1, n2, n3) in enumerate(delaunay_graph.neighbors):
        if triangle_index in free_indices_set:
            segments_graph.add_node(triangle_index, vertex=voronoi_vertices[triangle_index, :], radius=voronoi_radii[triangle_index])
            if n1 < triangle_index and n1 != -1 and n1 in free_indices_set:
                segments_graph.add_edge(triangle_index, int(n1))
            if n2 < triangle_index and n2 != -1 and n2 in free_indices_set:
                segments_graph.add_edge(triangle_index, int(n2))
            if n3 < triangle_index and n3 != -1 and n3 in free_indices_set:
                segments_graph.add_edge(triangle_index, int(n3))

    print("plotting")

    if path.exists(voronoi_plot_file_path) and do_not_recompute:
        print_info("do_not_recompute: will not recompute the voronoi plot")
        return

    if not path.exists(path.dirname(voronoi_plot_file_path)):
        os.makedirs(path.dirname(voronoi_plot_file_path))

    fig, ax = plt.subplots()
    fig.set_size_inches(*cm_to_body_parts(10 * m.map_size_meters))

    # plot circles
    for _, node_data in segments_graph.nodes.data():
        ax.add_artist(plt.Circle(node_data['vertex'], node_data['radius'], color='grey', fill=False, linewidth=0.05))

    # plot segments
    for node_index, node_data in segments_graph.nodes.data():
        x_1, y_1 = voronoi_vertices[node_index, :]
        for neighbor_index in segments_graph.neighbors(node_index):
            if neighbor_index < node_index:
                x_2, y_2 = voronoi_vertices[neighbor_index, :]
                ax.plot((x_1, x_2), (y_1, y_2), color='black', linewidth=1.0)

    # plot chain vertices
    chain_nodes = list(filter(lambda n: len(list(segments_graph.neighbors(n))) == 3, segments_graph.nodes))
    chain_vertices = voronoi_vertices[chain_nodes, :]
    ax.scatter(chain_vertices[:, 0], chain_vertices[:, 1], color='blue', s=2.0, marker='o')

    # plot leaf vertices
    leaf_nodes = list(filter(lambda n: len(list(segments_graph.neighbors(n))) == 1, segments_graph.nodes))
    leaf_vertices = voronoi_vertices[leaf_nodes, :]
    ax.scatter(leaf_vertices[:, 0], leaf_vertices[:, 1], color='red', s=2.0, marker='o')

    # plot vertical and horizontal wall points
    ax.scatter(*zip(*v_wall_points), s=15.0, marker='_')
    ax.scatter(*zip(*h_wall_points), s=15.0, marker='|')

    fig.savefig(voronoi_plot_file_path)
    plt.close(fig)


if __name__ == '__main__':
    environment_folders = sorted(glob.glob(path.expanduser("~/ds/performance_modelling/dataset/airlab")))
    print_info("gridmap_to_mesh {}%".format(0))
    for progress, environment_folder in enumerate(environment_folders):
        print_info("gridmap_to_mesh {}% {}".format((progress + 1)*100//len(environment_folders), environment_folder))
        map_info_file_path = path.join(environment_folder, "data", "map.yaml")
        result_voronoi_plot_file_path = path.join(environment_folder, "data", "visualization", "voronoi.svg")
        gridmap_to_voronoi_graph(map_info_file_path, result_voronoi_plot_file_path)
