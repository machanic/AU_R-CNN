import config
from dataset_toolkit.compress_utils import get_zip_ROI_AU
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import cv2
from img_toolkit.face_landmark import FaceLandMark
import  numpy as np
from img_toolkit.geometry_utils import sort_clockwise
from img_toolkit.face_mask_cropper import FaceMaskCropper
from collections import OrderedDict


def generate_AU_ROI_table():
    already_AU_couple = set()
    AU_couple_dict = get_zip_ROI_AU()
    AU_couple_region = OrderedDict()
    for AU, region_numbers in sorted(config.AU_ROI.items(), key=lambda e:e[0]):
        AU_couple = AU_couple_dict[AU]
        if AU_couple not in already_AU_couple:
            already_AU_couple.add(AU_couple)
            AU_couple_region[AU_couple] = sorted(region_numbers)

    for AU_couple, region_numbers in AU_couple_region.items():
        AU_info = []
        for AU in AU_couple:
            AU_info.append("AU {}".format(AU))
        AU_couple = " , ".join(AU_info)
        region_numbers = " , ".join(map(str, region_numbers))
        print("{0} & {1} \\\\".format(AU_couple, region_numbers))
        print("\hline")