import random
from collections import defaultdict

from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from dataset_toolkit.compress_utils import get_zip_ROI_AU
from img_toolkit.face_mask_cropper import FaceMaskCropper
import os
import config
import numpy as np
from collections import defaultdict
def collect_box_coordinate(image_path, mc_manager, database):
    au_couple_dict = get_zip_ROI_AU()
    AU_group_box_coodinate = defaultdict(list)  # AU_group => box_list
    key_prefix = database + "@512|"
    try:
        cropped_face, AU_box_dict = FaceMaskCropper.get_cropface_and_box(image_path, image_path,
                                                                     channel_first=True,
                                                                     mc_manager=mc_manager, key_prefix=key_prefix)
    except IndexError:
        return AU_group_box_coodinate
    for AU, box_list in AU_box_dict.items():
        AU_couple = au_couple_dict[AU]
        if AU_couple in AU_group_box_coodinate:
            continue
        new_box_list = [list(box) for box in box_list]
        new_box_list.sort(key=lambda e:e[1])
        AU_group_box_coodinate[AU_couple].extend(new_box_list)
    return AU_group_box_coodinate


def read_idx_file(file_path_list, mc_cached):
    tot_AU_group_box = defaultdict(list)
    all_readline = []
    for file_path in file_path_list:
        print("processing {}".format(file_path))
        all_readline.extend(open(file_path, "r").readlines())
    random.shuffle(all_readline)
    for line in all_readline:
        path = line.split()[0]
        print("processing {}".format(path))
        database = line.split()[-1]
        abs_path = config.RGB_PATH[database] + "/" + path
        AU_group_box_coodinate = collect_box_coordinate(abs_path, mc_cached, database)
        for AU_couple, box_list in AU_group_box_coodinate.items():
            tot_AU_group_box[AU_couple].append(box_list)  # box_list = 2 or 1 in different AU_couple

    for AU_couple, box_list_list in tot_AU_group_box.items():
        boxes = np.asarray(box_list_list) #  N x 2 or N x 1
        boxes = np.mean(boxes, axis=0)  # 2 or 1
        print(AU_couple, boxes)

if __name__ == "__main__":
    database = "DISFA"
    file_path_list = ["/home/machen/dataset/{}/idx/3_fold/id_trainval_1.txt".format(database),
                      "/home/machen/dataset/{}/idx/3_fold/id_test_1.txt".format(database)]
    from collections_toolkit.memcached_manager import PyLibmcManager

    adaptive_AU_database(database)
    mc_manager = PyLibmcManager('127.0.0.1')

    read_idx_file(file_path_list, mc_manager)
    print("{} done!".format(database))