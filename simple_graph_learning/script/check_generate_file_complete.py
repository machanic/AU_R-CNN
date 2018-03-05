import os
import numpy as np
import config
from collections import defaultdict

def read_folder(folder, database):
    box_in_frame = config.BOX_NUM[database]
    seq_name_dict = defaultdict(int)
    for file_name in os.listdir(folder):
        seq_name = file_name[:file_name.rindex("@")]
        abs_name = folder + os.sep + file_name
        boxes = np.load(abs_name)['bbox']
        length = boxes.shape[0] // box_in_frame
        seq_name_dict[seq_name] += length
    return seq_name_dict



def walk_folder(folder, orig_seq_name_dict):
    for database_folder in os.listdir(folder):
        for train_test_folder in os.listdir(folder + os.sep + database_folder):
            seq_name_dict = read_folder(folder+os.sep + database_folder + os.sep + train_test_folder)
