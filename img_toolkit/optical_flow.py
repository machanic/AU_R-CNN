from functools import lru_cache

import numpy as np
import optical_flow.RLOF as RLOF

from config import READ_COLOR_TYPE
from img_toolkit.image_tools import get_real_img


def get_flow_mat_from_img(img1, img2):
    flow = RLOF.generate_flowmat(
        RLOF.Mat.from_array(img1),
        RLOF.Mat.from_array(img2))
    return np.asarray(flow)

@lru_cache(maxsize=128)
def get_flow_mat_cache(img_path1, img_path2, database):
    img1 = get_real_img(img_path1,read_color_type=READ_COLOR_TYPE[database])
    img2 = get_real_img(img_path2, read_color_type=READ_COLOR_TYPE[database])
    return get_flow_mat_from_img(img1,img2)