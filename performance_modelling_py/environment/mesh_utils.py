#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os
from os import path
from typing import Optional

import numpy as np
import collada as cd
from collada import source
from performance_modelling_py.environment.ground_truth_map_utils import GroundTruthMap
from performance_modelling_py.utils import print_info
from scipy import ndimage


class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.index = None

    @property
    def ls(self):
        return [self.x, self.y, self.z]


class Normal:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.index = None

    @property
    def ls(self):
        return [self.x, self.y, self.z]


class Triangle:
    def __init__(self, v_1, v_2, v_3, n_1, n_2, n_3):
        self.v_1 = v_1
        self.v_2 = v_2
        self.v_3 = v_3
        self.normal_1 = n_1
        self.normal_2 = n_2
        self.normal_3 = n_3

    @property
    def ls(self):
        return [self.v_1.index, self.normal_1.index, self.v_2.index, self.normal_2.index, self.v_3.index, self.normal_3.index]


def wall_face(x_1, y_1, x_2, y_2, f, h, orientation):
    vertices_list = [
        Vertex(x_1, y_1, f),
        Vertex(x_1, y_1, f + h),
        Vertex(x_2, y_2, f),
        Vertex(x_2, y_2, f + h),
    ]

    orientations = {
        'x': [1, 0, 0],
        '-x': [-1, 0, 0],
        'y': [0, 1, 0],
        '-y': [0, -1, 0],
    }

    wall_normal_0 = Normal(*orientations[orientation])
    wall_normal_1 = Normal(*orientations[orientation])
    wall_normal_2 = Normal(*orientations[orientation])
    wall_normal_3 = Normal(*orientations[orientation])
    normals_list = [wall_normal_0, wall_normal_1, wall_normal_2, wall_normal_3]

    if orientation in ['x', '-y']:
        triangles_list = [
            Triangle(vertices_list[2], vertices_list[1], vertices_list[0], wall_normal_2,  wall_normal_1,  wall_normal_0),
            Triangle(vertices_list[1], vertices_list[2], vertices_list[3], wall_normal_1,  wall_normal_2,  wall_normal_3),
        ]
    elif orientation in ['-x', 'y']:
        triangles_list = [
            Triangle(vertices_list[0], vertices_list[1], vertices_list[2], wall_normal_0, wall_normal_1, wall_normal_2),
            Triangle(vertices_list[3], vertices_list[2], vertices_list[1], wall_normal_3, wall_normal_2, wall_normal_1),
        ]
    else:
        raise KeyError("orientation not in ['x', '-y', '-x', 'y']")

    return vertices_list, normals_list, triangles_list


def wall_top(x_min, y_min, x_max, y_max, h):
    vertices_list = [
        Vertex(x_min, y_min, h),
        Vertex(x_min, y_max, h),
        Vertex(x_max, y_min, h),
        Vertex(x_max, y_max, h),
    ]

    wall_normal = Normal(0, 0, 1)
    normals_list = [wall_normal]

    triangles_list = [
        Triangle(vertices_list[2], vertices_list[1], vertices_list[0], wall_normal,  wall_normal,  wall_normal),
        Triangle(vertices_list[1], vertices_list[2], vertices_list[3], wall_normal,  wall_normal,  wall_normal),
    ]
    return vertices_list, normals_list, triangles_list


def gridmap_to_mesh(grid_map_info_file_path, mesh_file_path, do_not_recompute=False, map_floor_height=0.0, wall_height=2.0):

    if path.exists(mesh_file_path):
        if do_not_recompute:
            print_info("do_not_recompute: will not recompute the output mesh")
            return

    if not path.exists(path.dirname(mesh_file_path)):
        os.makedirs(path.dirname(mesh_file_path))

    m = GroundTruthMap(grid_map_info_file_path)
    pixels = m.map_image.load()

    # occupied_bitmap contains 1 where pixels are occupied (black color value -> wall)
    # occupied_bitmap coordinates have origin in the image bottom-left with y-up rather than top-left with y-down,
    # so it is in between image coordinates and map frame coordinates
    w, h = m.map_image.size
    occupied_bitmap = np.zeros((w+1, h+1), dtype=int)
    for y in range(h):
        for x in range(w):
            occupied_bitmap[x, h-1-y] = pixels[x, y] == (0, 0, 0)

    # span the image with kernels to find vertical walls, north -> +1, south -> -1
    vertical_edges = ndimage.convolve(occupied_bitmap, np.array([[1, -1]]), mode='constant', cval=0, origin=np.array([0, -1]))
    north_bitmap = vertical_edges == 1
    south_bitmap = vertical_edges == -1

    # span the image with kernels to find horizontal walls, west -> +1, east -> -1
    horizontal_edges = ndimage.convolve(occupied_bitmap, np.array([[1], [-1]]), mode='constant', cval=0, origin=np.array([-1, 0]))
    west_bitmap = horizontal_edges == 1
    east_bitmap = horizontal_edges == -1

    vertices = list()
    normals = list()
    triangles = list()

    # make the mesh for the north and south walls
    for y in range(north_bitmap.shape[1]):
        north_wall_start: Optional[np.array] = None
        south_wall_start: Optional[np.array] = None
        for x in range(north_bitmap.shape[0]):
            if north_wall_start is None and north_bitmap[x, y]:  # wall goes up
                north_wall_start = m.image_y_up_to_map_frame_coordinates(x, y)
            if north_wall_start is not None and not north_bitmap[x, y]:  # wall goes down
                north_wall_end = m.image_y_up_to_map_frame_coordinates(x, y)
                v, n, t = wall_face(*north_wall_start, *north_wall_end, map_floor_height, wall_height, '-y')
                vertices += v
                normals += n
                triangles += t
                north_wall_start = None

            if south_wall_start is None and south_bitmap[x, y]:  # wall goes up
                south_wall_start = m.image_y_up_to_map_frame_coordinates(x, y)
            if south_wall_start is not None and not south_bitmap[x, y]:  # wall goes down
                south_wall_end = m.image_y_up_to_map_frame_coordinates(x, y)
                v, n, t = wall_face(*south_wall_start, *south_wall_end, map_floor_height, wall_height, 'y')
                vertices += v
                normals += n
                triangles += t
                south_wall_start = None

    # make the mesh for the west and east walls
    for x in range(west_bitmap.shape[0]):
        west_wall_start: Optional[np.array] = None
        east_wall_start: Optional[np.array] = None
        for y in range(west_bitmap.shape[1]):
            if west_wall_start is None and west_bitmap[x, y]:  # wall goes up
                west_wall_start = m.image_y_up_to_map_frame_coordinates(x, y)
            if west_wall_start is not None and not west_bitmap[x, y]:  # wall goes down
                west_wall_end = m.image_y_up_to_map_frame_coordinates(x, y)
                v, n, t = wall_face(*west_wall_start, *west_wall_end, map_floor_height, wall_height, '-x')
                vertices += v
                normals += n
                triangles += t
                west_wall_start = None

            if east_wall_start is None and east_bitmap[x, y]:  # wall goes up
                east_wall_start = m.image_y_up_to_map_frame_coordinates(x, y)
            if east_wall_start is not None and not east_bitmap[x, y]:  # wall goes down
                east_wall_end = m.image_y_up_to_map_frame_coordinates(x, y)
                v, n, t = wall_face(*east_wall_start, *east_wall_end, map_floor_height, wall_height, 'x')
                vertices += v
                normals += n
                triangles += t
                east_wall_start = None

    # make the mesh for the top of the walls
    for y in range(occupied_bitmap.shape[1]):
        x_min, y_min = None, None
        for x in range(occupied_bitmap.shape[0]):
            if x_min is None and occupied_bitmap[x, y]:  # wall goes up
                x_min, y_min = x, y
            if x_min is not None and not occupied_bitmap[x, y]:  # wall goes down
                # find the rectangle covering the most wall by checking each horizontal line
                x_max = x
                y_max = None
                subbitmap = occupied_bitmap[x_min:x_max, y_min:h+1]
                for square_h in range(subbitmap.shape[1]):
                    if not np.all(subbitmap[:, square_h]):
                        y_max = y_min + square_h
                        break
                # once found the square, delete the corresponding pixels (so they won't be checked again)
                occupied_bitmap[x_min:x_max, y_min:y_max] = 0
                v, n, t = wall_top(*m.image_y_up_to_map_frame_coordinates(x_min, y_min),
                                   *m.image_y_up_to_map_frame_coordinates(x_max, y_max), wall_height)
                vertices += v
                normals += n
                triangles += t
                x_min = None

    mesh = cd.Collada()
    effect = cd.material.Effect("effect0", [], "phong", diffuse=(1, 0, 0), specular=(0, 1, 0))
    mat = cd.material.Material("material0", "mymaterial", effect)
    mesh.effects.append(effect)
    mesh.materials.append(mat)

    cd_vertices_list = list()
    for index, vertex in enumerate(vertices):
        cd_vertices_list += vertex.ls
        vertex.index = index

    cd_normals_list = list()
    for index, normal in enumerate(normals):
        cd_normals_list += normal.ls
        normal.index = index

    cd_triangles_list = list()
    for triangle in triangles:
        cd_triangles_list += triangle.ls

    vert_src = cd.source.FloatSource("cubeverts-array", np.array(cd_vertices_list), ('X', 'Y', 'Z'))
    normal_src = cd.source.FloatSource("cubenormals-array", np.array(cd_normals_list), ('X', 'Y', 'Z'))
    geom = cd.geometry.Geometry(mesh, "geometry0", "mycube", [vert_src, normal_src])

    input_list = cd.source.InputList()
    input_list.addInput(0, 'VERTEX', "#cubeverts-array")
    input_list.addInput(1, 'NORMAL', "#cubenormals-array")

    triset = geom.createTriangleSet(np.array(cd_triangles_list), input_list, "materialref")
    geom.primitives.append(triset)
    mesh.geometries.append(geom)

    matnode = cd.scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = cd.scene.GeometryNode(geom, [matnode])
    node = cd.scene.Node("node0", children=[geomnode])
    myscene = cd.scene.Scene("myscene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene
    mesh.write(mesh_file_path)


if __name__ == '__main__':
    environment_folders = sorted(glob.glob(path.expanduser("~/ds/performance_modelling/dataset/*")))
    print_info("gridmap_to_mesh {}%".format(0))
    for progress, environment_folder in enumerate(environment_folders):
        print_info("gridmap_to_mesh {}% {}".format((progress + 1)*100//len(environment_folders), environment_folder))
        map_info_file_path = path.join(environment_folder, "data", "map.yaml")
        result_mesh_file_path = path.join(environment_folder, "data", "meshes", "extruded_map.dae")
        gridmap_to_mesh(map_info_file_path, result_mesh_file_path)
