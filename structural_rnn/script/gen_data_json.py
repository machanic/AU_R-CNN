import os
import numpy as np
import json
def gen_data_json(train_folder, test_folder):
    non_zero_feature_idx = set()
    label_set = set()
    out_folder = os.path.dirname(train_folder)
    for file_name in os.listdir(train_folder):
        print(file_name)
        file_path = train_folder + os.sep + file_name
        with open(file_path, "r") as file_obj:
            for line in file_obj:
                if line.startswith("#edge"): continue
                lines = line.split()
                label = lines[1]
                label_set.add(label)
                feature = np.asarray(list(map(float,lines[2][len("features:"):].split(","))))
                non_zero_feature_idx.update(list(map(int,np.nonzero(feature)[0])))
    for file_name in os.listdir(test_folder):
        print(file_name)
        file_path = test_folder + os.sep + file_name
        with open(file_path, "r") as file_obj:
            for line in file_obj:
                if line.startswith("#edge"): continue
                lines = line.split()
                feature = np.asarray(list(map(float, lines[2][len("features:"):].split(","))))
                non_zero_feature_idx.update(list(map(int, np.nonzero(feature)[0])))
    out_path = out_folder + os.sep + "data_info.json"
    data_info = {"non_zero_attrib_index": sorted(non_zero_feature_idx), "num_label":len(label_set)}
    with open(out_path, "w") as file_obj:
        file_obj.write(json.dumps(data_info))
        file_obj.flush()

if __name__ == "__main__":
    gen_data_json("/home/machen/dataset/toy/train","/home/machen/dataset/toy/test")