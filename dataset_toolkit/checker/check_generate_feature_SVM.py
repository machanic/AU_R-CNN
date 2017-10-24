import numpy as np
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
# this script is used for check whether the AU R_CNN extract feature is OK.

folder = "./"
X = []
Y = []
label_dict = dict() # label_str => label_int
attrib = dict()
trn_filename = ["F002_T1.txt", "M018_T3.txt", "M018_T7.txt"]
root_dir = "/home/machen/face_expr/result/graph/BP4D_3_fold_1/train/"
file_names = list(filter(lambda e: e.endswith(".txt") and e in trn_filename, os.listdir(root_dir)))
for file_name in file_names:
    with open(root_dir+os.sep+file_name, "r") as file_obj:
        for line in file_obj:
            if line.startswith("#edge"):
                continue
            lines = line.strip().split()
            label_str = lines[1]
            if label_str not in label_dict:
                label_dict[label_str] = len(label_dict)
            label_str = label_str[1:-1]
            feature = lines[2][len("features:"):]
            feature = np.array(list(map(float, feature.split(","))),dtype=np.float32)

            y = np.array(list(map(int, label_str.split(","))))

            Y.append(y)
            X.append(feature)
X = np.asarray(X)
Y = np.array(Y)
YT= np.transpose(Y)
for idx, y in enumerate(YT):
    if not np.any(y):
        print("idx {} not occur!".format(idx))


classif = OneVsRestClassifier(SVC(kernel='linear'))
print("classify train begin")
classif.fit(X,Y)
print("classify train over")
X = []

Y = []
file_name = "/home/machen/face_expr/result/graph/BP4D_3_fold_1/valid/M017_T1.txt"
with open(file_name, "r") as file_obj:
    for line in file_obj:
        if line.startswith("#edge"):
            continue
        lines = line.strip().split()
        label_str = lines[1]
        if label_str not in label_dict:
            label_dict[label_str] = len(label_dict)
        label_str = label_str[1:-1]
        feature = lines[2][len("features:"):]
        feature = np.array(list(map(float, feature.split(","))), dtype=np.float32)

        y = np.array(list(map(int, label_str.split(","))))

        Y.append(y)
        X.append(feature)
gt_Y = np.array(Y)

X = np.array(X)



pred_Y = classif.predict(X)
score = f1_score(gt_Y,pred_Y , average='weighted')
print(score)  # this score is for check generate feature is valid feature
