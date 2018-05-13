from collections import defaultdict
import os
import numpy as np
import config
from itertools import groupby
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database

def read_idx_file(file_path):
    label_matrix = defaultdict(list)
    with open(file_path, "r") as file_obj:
        for line in file_obj:
            seq_key = line.split()[0][:line.split()[0].rindex("/")]
            AUs = [AU for AU in line.split()[1].split(",")]
            each_frame_label = np.zeros(len(config.AU_SQUEEZE), dtype=np.int32)
            for AU in AUs:
                if AU == "0":continue
                AU_idx = config.AU_SQUEEZE.inv[AU]
                each_frame_label[AU_idx] = 1
            label_matrix[seq_key].append(each_frame_label)
    for seq_key, frame_label_list in label_matrix.items():
        label_matrix[seq_key] = np.stack(frame_label_list, axis=0)
    return label_matrix

def stats_frequency(label_matrix_dict):
    AU_all_count = defaultdict(list)
    AU_segment_count = defaultdict(int)
    for key, label_matrix in label_matrix_dict.items():  # N x AU
        label_matrix = label_matrix.transpose()  # AU x N
        for AU_idx, column in enumerate(label_matrix):
            AU_continous_count = defaultdict(list)
            for label, group in groupby(column):
                if label == 1:
                    AU_segment_count[config.AU_SQUEEZE[AU_idx]] += 1
                AU_continous_count[label].append(sum(1 for _ in group))
            if 0 in AU_continous_count:
                del AU_continous_count[0]  # only have 1
            else:
                pass
            for label, val_list in AU_continous_count.items():
                for sum_val in val_list:
                    AU_all_count[config.AU_SQUEEZE[AU_idx]].append(sum_val)
    average_dict = {}
    for AU, sum_val_list in AU_all_count.items():
        average_dict[AU] = sum(sum_val_list)/len(sum_val_list)
    return average_dict, AU_segment_count


if __name__ == "__main__":
    adaptive_AU_database("BP4D", False)
    label_matrix_dict =read_idx_file("/home/machen/dataset/BP4D/idx/3_fold/id_trainval_1.txt")
    average_dict, AU_segment_count = stats_frequency(label_matrix_dict)
    for AU, mean in sorted(average_dict.items(), key=lambda e:e[1]):
        if AU in config.paper_use_BP4D:
            print(AU, mean)

    for AU, seg_count in sorted(AU_segment_count.items(), key=lambda e:e[1]):
        if AU in config.paper_use_BP4D:
            print(AU, seg_count)