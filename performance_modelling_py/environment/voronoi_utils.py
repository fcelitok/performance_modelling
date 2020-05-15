#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
from os import path
import numpy as np
from performance_modelling_py.environment.ground_truth_map_utils import GroundTruthMap
from performance_modelling_py.utils import print_info
from scipy import ndimage
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


def gridmap_to_voronoi_graph(grid_map_info_file_path, voronoi_graph_file_path, do_not_recompute=False):

    if path.exists(voronoi_graph_file_path):
        if do_not_recompute:
            print_info("do_not_recompute: will not recompute the output voronoi graph")
            return

    m = GroundTruthMap(grid_map_info_file_path)

    occupied_bitmap, north_bitmap, south_bitmap, west_bitmap, east_bitmap = m.edge_bitmaps(lambda pixel: pixel != m.free_rgb)

    v_corners = set()
    h_corners = set()

    w, h = north_bitmap.shape
    for x in range(w):
        for y in range(h):
            if north_bitmap[x, y] or south_bitmap[x, y]:
                v_corners.add((x, y))
                v_corners.add((x+1, y))
            if west_bitmap[x, y] or east_bitmap[x, y]:
                h_corners.add((x, y))
                h_corners.add((x, y+1))

    corners = v_corners.union(h_corners)
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

    free_indices = list(np.where(1 - voronoi_vertices_occupancy_bitmap)[0])
    free_indices_set = set(list(np.where(1 - voronoi_vertices_occupancy_bitmap)[0]))

    free_segments = list()
    segments_graph = nx.Graph()

    for triangle_index, (n1, n2, n3) in enumerate(delaunay_graph.neighbors):
        if triangle_index not in free_indices_set:
            continue
        center = voronoi_vertices[triangle_index, :]
        if n1 < triangle_index and n1 != -1 and n1 in free_indices_set:
            center_neighbor_1 = voronoi_vertices[n1, :]
            free_segments.append((center, center_neighbor_1))
            segments_graph.add_edge(triangle_index, int(n1))
        if n2 < triangle_index and n2 != -1 and n2 in free_indices_set:
            center_neighbor_2 = voronoi_vertices[n2, :]
            free_segments.append((center, center_neighbor_2))
            segments_graph.add_edge(triangle_index, int(n2))
        if n3 < triangle_index and n3 != -1 and n3 in free_indices_set:
            center_neighbor_3 = voronoi_vertices[n3, :]
            free_segments.append((center, center_neighbor_3))
            segments_graph.add_edge(triangle_index, int(n3))

    chain_nodes = list(filter(lambda n: len(list(segments_graph.neighbors(n))) != 2, segments_graph.nodes))

    for chain_node in chain_nodes:
        print(chain_node)
    chain_free_vertices = voronoi_vertices[chain_nodes, :]
    print(chain_free_vertices)

    free_voronoi_vertices = voronoi_vertices[free_indices, :]
    free_voronoi_radii = voronoi_radii[free_indices]

    print("plotting")
    fig, ax = plt.subplots()
    fig.set_size_inches(*cm_to_body_parts(40, 40))

    for c, r in zip(free_voronoi_vertices, free_voronoi_radii):
        circle = plt.Circle(c, r, color='grey', fill=False, linewidth=0.05)
        ax.add_artist(circle)

    for (x1, y1), (x2, y2) in free_segments:
        ax.plot((x1, x2), (y1, y2), color='black', linewidth=0.25)

    ax.scatter(chain_free_vertices[:, 0], chain_free_vertices[:, 1], s=1.0, marker='o')
    ax.scatter(*zip(*v_corners), s=.1, marker='|')
    ax.scatter(*zip(*h_corners), s=.1, marker='_')

    figure_output_path = path.expanduser("~/tmp/voronoi.svg")
    fig.savefig(figure_output_path)
    plt.close(fig)
    # voronoi_graph.write(voronoi_graph_file_path)


if __name__ == '__main__':
    environment_folders = sorted(glob.glob(path.expanduser("~/ds/performance_modelling/dataset/airlab")))
    print_info("gridmap_to_mesh {}%".format(0))
    for progress, environment_folder in enumerate(environment_folders):
        print_info("gridmap_to_mesh {}% {}".format((progress + 1)*100//len(environment_folders), environment_folder))
        map_info_file_path = path.join(environment_folder, "data", "map.yaml")
        result_voronoi_graph_file_path = path.join(environment_folder, "data", "graphs", "voronoi_graph.???")
        gridmap_to_voronoi_graph(map_info_file_path, result_voronoi_graph_file_path)
