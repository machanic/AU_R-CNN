from collections import defaultdict
import os
import numpy as np
import math

def check_label_balance(folder_path):

    counter = defaultdict(int)
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"): continue
        print(filename)
        with open(folder_path+os.sep + filename, "r") as file_obj:
            for line in file_obj:
                if line.startswith("#edge"): continue
                line = line.rstrip()
                label = line.split()[1]
                label_bin = np.array(list(map(int, label[1:-1].split(','))),dtype=np.int32)
                if not np.any(label_bin):
                    continue
                counter[label] += 1
    return counter

def make_dict(path, counter):
    max_count = max(counter.values())
    with open(path, "w") as file_obj:
        for label, count  in counter.items():
            label = label[1:-1]
            file_obj.write("{0} {1}\n".format(label, round(max_count/float(count))))
        file_obj.flush()

if __name__ == "__main__":
    label_counter  = check_label_balance("/home/machen/face_expr/result/graph_backup")
    make_dict("/home/machen/face_expr/result/graph_backup/label_imbalance.txt", label_counter)