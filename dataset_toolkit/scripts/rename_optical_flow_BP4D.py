from collections import defaultdict
import os
import shutil

def read_file_path(folder):
    file_dict = defaultdict(list)
    for subject_name in os.listdir(folder):
        for seq_key in os.listdir(folder + "/" + subject_name):
            for img_name in os.listdir(folder + "/" + subject_name + "/" + seq_key):
                img_no = img_name[:img_name.rindex(".")]
                file_dict[subject_name + "/" + seq_key].append(img_no)
    return file_dict

def compare_rename(trn_dir, of_dir):
    trn_dict = read_file_path(trn_dir)
    of_dict = read_file_path(of_dir)
    new_of_dir = of_dir + "_new"

    for seq_key, orig_list in trn_dict.items():
        of_list = of_dict[seq_key]
        if len(of_list) + 1 != len(orig_list):
            print(seq_key)
        of_list = sorted(of_list, key=lambda e:int(e))
        orig_list = sorted(orig_list,key=lambda e:int(e))
        copy=False
        assert len(orig_list) == len(of_list) + 1
        for idx, orig_no in enumerate(orig_list):
            if idx == len(of_list):
                of_name = of_list[idx-1] + ".jpg"

            else:
                of_name = of_list[idx] + ".jpg"

            if idx >= len(of_list) - 1:
                copy=True

            of_path = of_dir + "/" + seq_key + "/" + of_name
            new_path = new_of_dir + "/" + seq_key + "/" + orig_no + ".jpg"
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            if not copy:
                shutil.move(of_path, new_path)
            else:
                shutil.copyfile(of_path, new_path)
                print("copy from {0} to {1}".format(of_path, new_path))

if __name__ == "__main__":
    compare_rename("/home/machen/dataset/BP4D/BP4D-training", "/home/machen/dataset/BP4D/optical_flow")