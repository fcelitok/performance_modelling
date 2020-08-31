#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import os
import sys
from os import path
import cv2
import numpy as np
import rospy
import yaml
import rosbag


def save_map_image(original_map_msg, image_file_path, width, height, info_file_path=None, map_free_threshold=25, map_occupied_threshold=65):
    map_msg = copy.deepcopy(original_map_msg)
    map_image = np.full(fill_value=205, shape=(height, width), dtype=np.uint8)

    for y in range(map_msg.info.height):
        for x in range(map_msg.info.width):
            i = x + (map_msg.info.height - y - 1) * map_msg.info.width
            if 0 <= map_msg.data[i] <= map_free_threshold:  # [0, free)
                map_image[y, x] = 254
            elif map_msg.data[i] >= map_occupied_threshold:  # (occ, 255]
                map_image[y, x] = 0
            else:  # [free, occ]
                map_image[y, x] = 205

    # save map image
    cv2.imwrite(image_file_path, map_image)

    # save map info
    if info_file_path is not None:
        with open(info_file_path, 'w') as yaml_file:
            yaml_dict = {
                'header': {
                    'seq': map_msg.header.seq,
                    'stamp': map_msg.header.stamp.to_sec(),
                    'frame_id': map_msg.header.frame_id,
                    },
                'info': {
                    'map_load_time': map_msg.info.map_load_time.to_sec(),
                    'resolution': map_msg.info.resolution,
                    'width': map_msg.info.width,
                    'height': map_msg.info.height,
                    'origin': {
                        'position': {'x': map_msg.info.origin.position.x, 'y': map_msg.info.origin.position.y, 'z': map_msg.info.origin.position.z},
                        'orientation': {'x': map_msg.info.origin.orientation.x, 'y': map_msg.info.origin.orientation.y, 'z': map_msg.info.origin.orientation.z, 'w': map_msg.info.origin.orientation.w},
                    },
                },
            }
            yaml.dump(yaml_dict, yaml_file, default_flow_style=False)


if len(sys.argv) != 2:
    print("ARGS!")
    sys.exit()

bag_path = path.expanduser(sys.argv[1])
bag = rosbag.Bag(bag_path)

destination_dir = "./map_images/"
if not path.exists(destination_dir):
    os.makedirs(destination_dir)

i = 0
map_msg_list = list()
for topic, msg, t in bag.read_messages(topics=['/map']):
    map_msg_list.append(msg)

max_width = 0
max_height = 0
for msg in map_msg_list:
    if msg.info.width > max_width:
        max_width = msg.info.width
    if msg.info.height > max_height:
        max_height = msg.info.height

for msg in map_msg_list:
    map_image_path = path.join(destination_dir, "map_image_{i:05d}.png".format(i=i))
    print("[{t}] saving image to".format(t=t.to_sec()), map_image_path)
    save_map_image(msg, map_image_path, max_width, max_height)
    i += 1

bag.close()

# then use the following command to make a gif:
# ffmpeg -r 2 -i map_images/map_image_%05d.png map.gif
