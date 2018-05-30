import shutil
import os
import re
from collections import defaultdict


def read_file_path(folder):
    file_dict = defaultdict(list)
    for subject_name in os.listdir(folder):
        for img_name in os.listdir(folder + "/" + subject_name + "/"):
            img_no = img_name[:img_name.rindex(".")]
            file_dict[subject_name].append(img_no)
    return file_dict

if __name__ == "__main__":
    left_dict = read_file_path("/home/machen/dataset/DISFA/Img_LeftCamera")
    right_dict = read_file_path("/home/machen/dataset/DISFA/Img_RightCamera")
    new_dir = "/home/machen/dataset/DISFA/new_of/"

    pattern = re.compile("(.*?)Video(SN.*?)_.*")
    of_left = defaultdict(list)
    of_right = defaultdict(list)
    for dir_name in os.listdir("/home/machen/dataset/DISFA/optical_flow"):
        seq_ma = pattern.match(dir_name)
        seq_key = seq_ma.group(2)
        left_right = seq_ma.group(1)
        of_dict = of_left
        if left_right == "Right":
            of_dict = of_right

        for of_path in os.listdir("/home/machen/dataset/DISFA/optical_flow/" + dir_name):
            of_no = of_path[:of_path.rindex(".")]
            of_dict[seq_key].append(of_no)

    for img_dict in [left_dict, right_dict]:
        if img_dict == left_dict:
            continue
        of_dict = of_left
        if img_dict == right_dict:
            of_dict = of_right
        for seq_key, img_list in img_dict.items():
            of_list = of_dict[seq_key]
            img_list = sorted(img_list, key=lambda e:int(e))
            of_list = sorted(of_list, key=lambda e:int(e))
            if len(img_list) - 1 != len(of_list):
                print(len(img_list) - 1, len(of_list))
            # assert len(left_list) - 1 == len(left_of_list), seq_key
            for idx, img_orig in enumerate(img_list):
                if idx >= len(of_list):
                    idx = len(of_list) - 1
                src_no = of_list[idx]

                target_keystr = "Img_LeftCamera"
                keystr = "LeftVideo"
                if img_dict == right_dict:
                    keystr = "RightVideo"
                    target_keystr = "Img_RightCamera"
                src_path = "/home/machen/dataset/DISFA/optical_flow/{}".format(keystr) + seq_key+"_comp/" + src_no + ".jpg"

                target_path = new_dir + target_keystr +"/" + seq_key + "/" + img_orig + ".jpg"
                # os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # shutil.copyfile(src_path, target_path)
                print(" move {0} to {1}".format(src_path, target_path))

