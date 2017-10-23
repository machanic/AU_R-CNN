# -*- coding: utf-8 -*-
# Created by liangzh0707 on 2017/3/9
from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=128)
def get_real_flow(abs_path):
    flow_mat = np.load(abs_path)
    return flow_mat

@lru_cache(maxsize=2048)
def get_real_img(abs_path, read_color_type):
    img = cv2.imread(abs_path, read_color_type)
    img = cv2.resize(img, IMG_SIZE)
    return img


def draw_trajectory(images, locations, box_width, box_height, rgb_color, thick, max_prop_idx_ls):
    for idx, image in enumerate(images):
        center_locs = locations[idx]
        for (point_a_idx, point_b_idx) in zip(range(0, len(center_locs)-1), range(1, len(center_locs))):
            point_a, point_b = tuple(center_locs[point_a_idx]), tuple(center_locs[point_b_idx])
            point_a = (point_a[1], point_a[0])  # FIXME strange??? must y first
            point_b = (point_b[1], point_b[0])
            cv2.arrowedLine(image, point_a, point_b, rgb_color, thickness=thick)
            if point_b_idx == max_prop_idx_ls[idx]:
                cv2.rectangle(image,(int(point_b[0] - box_width/2), int(point_b[1] - box_height/2)),
                              (int(point_b[0] + box_width / 2), int(point_b[1] + box_height / 2)), rgb_color,
                              thickness=thick)



def draw_rectangle(images, center_locs, box_width, box_height, rgb_color,  thick):

    for idx, image in enumerate(images):
        cv2.rectangle(image, (int(center_locs[idx][0] - box_width/2), int(center_locs[idx][1] - box_height/2)),
                      (int(center_locs[idx][0] + box_width/2), int(center_locs[idx][1] + box_height/2)), rgb_color,
                      thickness=thick)