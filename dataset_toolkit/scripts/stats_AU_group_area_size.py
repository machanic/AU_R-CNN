from collections import defaultdict

from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from dataset_toolkit.compress_utils import get_zip_ROI_AU
from img_toolkit.face_mask_cropper import FaceMaskCropper
import os
import config
def stats_AU_group_area(image_path, mc_manager, database):
    au_couple_dict = get_zip_ROI_AU()
    AU_group_box_area = dict()
    key_prefix = database + "|"
    try:
        cropped_face, AU_box_dict = FaceMaskCropper.get_cropface_and_box(image_path, image_path,
                                                                     channel_first=True,
                                                                     mc_manager=mc_manager, key_prefix=key_prefix)
    except IndexError:
        return AU_group_box_area
    for AU, box_list in AU_box_dict.items():
        AU_couple = au_couple_dict[AU]
        tot_area = 0.
        for box in box_list:
            y_min, x_min, y_max, x_max = box
            area  = (x_max - x_min) * (y_max - y_min)
            tot_area += area
        tot_area /= len(box_list)
        AU_group_box_area[AU_couple] = tot_area
    return AU_group_box_area


def read_idx_file(file_path_list, mc_cached):
    all_AU_group = defaultdict(list)
    for file_path in file_path_list:
        print("processing {}".format(file_path))
        with open(file_path, "r") as file_obj:
            for line in file_obj:
                path = line.split()[0]
                print("processing {}".format(path))
                database = line.split()[-1]
                abs_path = config.RGB_PATH[database] + "/" + path
                AU_group_box_area = stats_AU_group_area(abs_path, mc_cached, database)
                for AU_couple, area in AU_group_box_area.items():

                    all_AU_group[AU_couple].append(area)

    for AU_couple, area_list in all_AU_group.items():
        print(AU_couple, sum(area_list)/ len(area_list))

if __name__ == "__main__":
    database = "BP4D"
    file_path_list = ["/home/machen/dataset/{}/idx/3_fold/id_trainval_1.txt".format(database),
                      "/home/machen/dataset/{}/idx/3_fold/id_test_1.txt".format(database)]
    from collections_toolkit.memcached_manager import PyLibmcManager

    adaptive_AU_database(database)
    mc_manager = PyLibmcManager('127.0.0.1')

    read_idx_file(file_path_list, mc_manager)