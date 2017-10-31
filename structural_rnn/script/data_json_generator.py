import os
import numpy as np
import json
import config
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database

def gen_data_json(train_folder, test_folder, label_dict): # pick_idx_dict key=choice idx value=AU
    out_folder = os.path.dirname(train_folder)
    num_attrib_type = 0
    for file_name in os.listdir(train_folder):

        file_path = train_folder + os.sep + file_name
        if file_path.endswith("npy"):
            h_info_array = np.load(file_path)
            num_attrib_type = h_info_array.shape[1]
            break



    out_path = out_folder + os.sep + "data_info.json"
    data_info = {"num_attrib_type": num_attrib_type,
                 "label_dict":label_dict}
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
        pick_idx_dict[config.AU_SQUEEZE.inv[AU]] = AU  # AU_idx -> AU

    gen_data_json("/home/machen/face_expr/result/graph/BP4D_3_fold_1/train","/home/machen/S_RNN_plus/result/graph/BP4D_toy/valid", pick_idx_dict)