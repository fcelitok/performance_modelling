#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from os import path
from typing import Optional

import numpy as np
import collada as cd
import yaml
from PIL import Image, ImageFilter
from collada import source
from performance_modelling_py.utils import print_info, print_error
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


def color_diff(a, b):
    return np.sum(np.array(b) - np.array(a)) / len(a)


def color_abs_diff(a, b):
    return np.abs(color_diff(a, b))


def rgb_less_than(a, b):
    return a[0] < b[0] and a[1] < b[1] and a[2] < b[2]


def cm_to_body_parts(*argv):
    inch = 2.54
    if isinstance(argv[0], tuple):
        return tuple(x_cm / inch for x_cm in argv[0])
    else:
        return tuple(x_cm / inch for x_cm in argv)


def gridmap_to_mesh(grid_map_file_path, grid_map_info_file_path, mesh_file_path, do_not_recompute=False, blur_filter_radius=1, occupied_threshold=205, wall_height=2.0):

    if path.exists(mesh_file_path):
        print_info("file already exists: {}".format(mesh_file_path))
        if do_not_recompute:
            print_info("do_not_recompute: will not recompute the output mesh")
            return

    map_image = Image.open(grid_map_file_path)
    print_info("opened image, mode: {mode}, image: {im}".format(mode=map_image.mode, im=grid_map_file_path))

    if map_image.mode != 'RGB':
        print('image mode is {mode} ({size}×{ch_num}), converting to RGB'.format(mode=map_image.mode, size=map_image.size, ch_num=len(map_image.split())))
        # remove alpha channel by pasting on white background
        background = Image.new("RGB", map_image.size, (254, 254, 254))
        background.paste(map_image)
        # swap the new image and apply a median filter
        map_image = background.filter(ImageFilter.BoxBlur(blur_filter_radius))
        file_path = path.splitext(grid_map_file_path)[0]
        map_image.save(f"{file_path}_filtered.pgm")

    pixels = map_image.load()

    # create a bitmap of occupied cells
    occupied_bitmap = np.zeros((map_image.size[0]+1, map_image.size[1]+1), dtype=int)
    w, h = map_image.size
    for y in range(h):
        for x in range(w):
            occupied_bitmap[x, h-1-y] = rgb_less_than(pixels[x, y], (occupied_threshold, occupied_threshold, occupied_threshold))  # pixels < (150, 150, 150) -> black -> occupied -> occupied_bitmap=1

    # span the image with kernels to find vertical walls, north -> +1, south -> -1
    vertical_edges = ndimage.convolve(occupied_bitmap, np.array([[1, -1]]), mode='constant', cval=0, origin=np.array([0, -1]))
    north_bitmap = vertical_edges == 1
    south_bitmap = vertical_edges == -1

    # span the image with kernels to find horizontal walls, west -> +1, east -> -1
    horizontal_edges = ndimage.convolve(occupied_bitmap, np.array([[1], [-1]]), mode='constant', cval=0, origin=np.array([-1, 0]))
    west_bitmap = horizontal_edges == 1
    east_bitmap = horizontal_edges == -1

    with open(grid_map_info_file_path, 'r') as info_file:
        info_yaml = yaml.load(info_file)

    if info_yaml['map']['pose']['theta'] != 0:
        print_error("gridmap_to_mesh: map rotation not supported")
        return

    map_size_meters = np.array([float(info_yaml['map']['size']['x']), float(info_yaml['map']['size']['y'])])
    print("map_size (meters):", map_size_meters)
    map_size_pixels = np.array(map_image.size)
    print("map_size (pixels):", map_size_pixels)
    resolution = map_size_meters / map_size_pixels * np.array([1, 1])  # meter/pixel, on both axis, except y axis is inverted in image
    print("resolution:", resolution)

    map_floor_height = float(info_yaml['map']['pose']['z'])
    map_offset_meters = np.array([float(info_yaml['map']['pose']['x']), float(info_yaml['map']['pose']['y'])])
    print("map_offset (meters):", map_offset_meters)

    # pixels_to_world_coordinates
    def w(x_pixels, y_pixels):
        p_pixels = np.array([x_pixels, y_pixels])
        return resolution * p_pixels - 0.5 * map_size_meters + map_offset_meters

    vertices = list()
    normals = list()
    triangles = list()

    for y in range(north_bitmap.shape[1]):
        north_wall_start: Optional[np.array] = None
        south_wall_start: Optional[np.array] = None
        for x in range(north_bitmap.shape[0]):
            if north_wall_start is None and north_bitmap[x, y]:  # wall goes up
                north_wall_start = w(x, y)
            if north_wall_start is not None and not north_bitmap[x, y]:  # wall goes down
                north_wall_end = w(x, y)
                v, n, t = wall_face(*north_wall_start, *north_wall_end, map_floor_height, wall_height, '-y')
                vertices += v
                normals += n
                triangles += t
                north_wall_start = None

            if south_wall_start is None and south_bitmap[x, y]:  # wall goes up
                south_wall_start = w(x, y)
            if south_wall_start is not None and not south_bitmap[x, y]:
                south_wall_end = w(x, y)
                v, n, t = wall_face(*south_wall_start, *south_wall_end, map_floor_height, wall_height, 'y')
                vertices += v
                normals += n
                triangles += t
                south_wall_start = None

    for x in range(west_bitmap.shape[0]):
        west_wall_start: Optional[np.array] = None
        east_wall_start: Optional[np.array] = None
        for y in range(west_bitmap.shape[1]):
            if west_wall_start is None and west_bitmap[x, y]:  # wall goes up
                west_wall_start = w(x, y)
            if west_wall_start is not None and not west_bitmap[x, y]:  # wall goes down
                west_wall_end = w(x, y)
                v, n, t = wall_face(*west_wall_start, *west_wall_end, map_floor_height, wall_height, '-x')
                vertices += v
                normals += n
                triangles += t
                west_wall_start = None

            if east_wall_start is None and east_bitmap[x, y]:  # wall goes up
                east_wall_start = w(x, y)
            if east_wall_start is not None and not east_bitmap[x, y]:
                east_wall_end = w(x, y)
                v, n, t = wall_face(*east_wall_start, *east_wall_end, map_floor_height, wall_height, 'x')
                vertices += v
                normals += n
                triangles += t
                east_wall_start = None

            # if west_bitmap[x, y]:
            #     # there is a west-facing wall to the west of this pixel
            #     v, n, t = wall_face(*w(x, y), *w(x, y + 1), map_floor_height, wall_height, '-x')
            #     vertices += v
            #     normals += n
            #     triangles += t
            # if east_bitmap[x, y]:
            #     # there is a east-facing wall to the west of this pixel
            #     v, n, t = wall_face(*w(x, y), *w(x, y + 1), map_floor_height, wall_height, 'x')
            #     vertices += v
            #     normals += n
            #     triangles += t

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

    print("cd_vertices_list", cd_vertices_list)
    print("cd_normals_list", cd_normals_list)
    print("cd_triangles_list", cd_triangles_list)

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

    environment_folder = path.expanduser("~/ds/performance_modelling_test_datasets/airlab/")
    map_file_path = path.join(environment_folder, "map.png")
    map_info_file_path = path.join(environment_folder, "stage_world_info.yaml")
    result_mesh_file_path = path.join(environment_folder, 'test.dae')
    gridmap_to_mesh(map_file_path, map_info_file_path, result_mesh_file_path)

    environment_folder = path.expanduser("~/ds/performance_modelling_test_datasets/test/")
    map_file_path = path.join(environment_folder, "map.png")
    map_info_file_path = path.join(environment_folder, "stage_world_info.yaml")
    result_mesh_file_path = path.join(environment_folder, 'test.dae')
    gridmap_to_mesh(map_file_path, map_info_file_path, result_mesh_file_path)

    environment_folder = path.expanduser("~/ds/performance_modelling_test_datasets/7A-2/")
    map_file_path = path.join(environment_folder, "map.png")
    map_info_file_path = path.join(environment_folder, "stage_world_info.yaml")
    result_mesh_file_path = path.join(environment_folder, 'test.dae')
    gridmap_to_mesh(map_file_path, map_info_file_path, result_mesh_file_path)
