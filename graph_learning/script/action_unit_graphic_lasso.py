
import os
import config
from sklearn.covariance import GraphLasso, GraphLassoCV
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
import numpy as np
from collections import defaultdict

def get_DISFA_prescion_matrix(label_file_dir):
    adaptive_AU_database("DISFA")
    alpha = 0.2
    model = GraphLassoCV(alphas=100,
                         cv=10,
                         max_iter=100,
                         tol=1e-5,
                         verbose=True,
                         mode="lars",
                         assume_centered=False, n_jobs=100)
    X = []
    for file_name in os.listdir(label_file_dir):
        subject_filename = label_file_dir + os.sep + file_name
        frame_label = defaultdict(dict)
        for au_file in os.listdir(subject_filename):
            abs_filename =  subject_filename + "/" + au_file
            AU = au_file[au_file.rindex("_")+3: au_file.rindex(".")]
            with open(abs_filename, "r") as file_obj:
                for line in file_obj:
                    frame, AU_label = line.strip().split(",")
                    # AU_label = int(AU_label)
                    AU_label = 0 if int(AU_label) < 3 else 1   # 居然<3的不要,但是也取得了出色的效果
                    frame_label[int(frame)][AU] = int(AU_label)
        for frame, AU_dict in frame_label.items():
            AU_bin = np.zeros(len(config.AU_SQUEEZE))
            for AU, AU_label in AU_dict.items():
                bin_idx = config.AU_SQUEEZE.inv[AU]
                np.put(AU_bin, bin_idx, AU_label)
            X.append(AU_bin)
    X = np.array(X)
    print(X.shape)
    model.fit(X)
    cov_ = model.covariance_
    prec_ = model.precision_
    return {"prec": prec_, "cov": cov_}

def get_BP4D_prescion_matrix(label_file_dir):
    adaptive_AU_database("BP4D")
    alpha = 0.2
    model = GraphLassoCV(alphas=100,
                         cv=10,
                       max_iter=10,
                       tol=1e-5,
                       verbose=True,
                         mode="lars",
                       assume_centered=False, n_jobs=100)

    X = []
    for file_name in os.listdir(label_file_dir): # each file is a video
        AU_column_idx = {}
        with open(label_file_dir + "/" + file_name, "r") as au_file_obj:  # each file is a video

            for idx, line in enumerate(au_file_obj):

                if idx == 0:  # header specify Action Unit
                    for col_idx, AU in enumerate(line.split(",")[1:]):
                        AU_column_idx[AU] = col_idx + 1  # read header
                    continue  # read head over , continue

                lines = line.split(",")
                frame = lines[0]
                au_labels = [AU for AU in config.AU_ROI.keys() \
                                 if int(lines[AU_column_idx[AU]]) == 1]
                AU_bin = np.zeros(len(config.AU_SQUEEZE))
                for AU in au_labels:
                    bin_idx = config.AU_SQUEEZE.inv[AU]
                    np.put(AU_bin, bin_idx, 1)
                X.append(AU_bin)
    X = np.array(X)
    print(X.shape)
    # X = np.transpose(X)
    model.fit(X)
    cov_ = model.covariance_
    prec_ = model.precision_

    return {"prec": prec_, "cov":cov_}

if __name__ == "__main__":
    ret_dict = get_DISFA_prescion_matrix(config.DATA_PATH["DISFA"] + "/ActionUnit_Labels/")
    # ret_dict = get_BP4D_prescion_matrix(config.DATA_PATH["BP4D"]+"/AUCoding/")
    cov, prec = ret_dict["cov"], ret_dict["prec"]
    print(cov.shape, prec.shape)
    related = set()
    for i,j in zip(*np.where(cov>0)):
        if i == j:continue
        related.add(tuple(sorted([int(config.AU_SQUEEZE[i]),int(config.AU_SQUEEZE[j])])))
    related_prec = set()
    for i,j in zip(*np.where(prec>0)):
        if i == j:continue
        related_prec.add(tuple(sorted([int(config.AU_SQUEEZE[i]),int(config.AU_SQUEEZE[j])])))
    print(related)
    print(related_prec)