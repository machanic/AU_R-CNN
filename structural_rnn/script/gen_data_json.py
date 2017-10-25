import os
import numpy as np
import json
import config
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database

def gen_data_json(train_folder, test_folder, pick_label_idx_dict): # pick_idx_dict key=choice idx value=AU
    non_zero_feature_idx = set()
    label_dict = dict()  # key is 0,1,1,1,0,0,0 value is int id
    out_folder = os.path.dirname(train_folder)

    all_choice_label_idx = list(sorted(pick_label_idx_dict.keys()))
    for file_name in os.listdir(train_folder):
        print(file_name)
        file_path = train_folder + os.sep + file_name
        with open(file_path, "r") as file_obj:
            for line in file_obj:
                if line.startswith("#edge"): continue
                lines = line.split()
                label = lines[1]
                label = np.asarray(label[1:-1].split(","), dtype=np.int32)
                assert len(label) >= max(pick_label_idx_dict.keys())
                label = label[all_choice_label_idx]
                label = ",".join(map(str, label))
                if label not in label_dict:
                    label_dict[label] = len(label_dict)

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
    data_info = {"non_zero_attrib_index": sorted(non_zero_feature_idx), "num_label":len(label_dict),
                 "use_label_idx":pick_label_idx_dict, "label_dict":label_dict}
    with open(out_path, "w") as file_obj:
        file_obj.write(json.dumps(data_info))
        file_obj.flush()

if __name__ == "__main__":

    database="BP4D"
    adaptive_AU_database(database)
    paper_use_AU = []
    if database == "BP4D":
        paper_use_AU = config.paper_use_BP4D
    elif database == "DISFA":
        paper_use_AU = config.paper_use_DISFA
    pick_idx_dict = {}
    for AU in paper_use_AU:
        pick_idx_dict[config.AU_SQUEEZE.inv[AU]] = AU

    gen_data_json("/home/machen/face_expr/result/graph/BP4D_3_fold_1/train","/home/machen/face_expr/result/graph/BP4D_3_fold_1/test", pick_idx_dict)