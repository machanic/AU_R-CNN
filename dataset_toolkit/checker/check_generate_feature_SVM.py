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
test_filename = "F001_T8.txt"
file_names = list(filter(lambda e: e.endswith(".txt") and e!=test_filename, os.listdir("./")))
for file_name in file_names:
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
            feature = np.array(list(map(float, feature.split(","))))
            x  = dict()
            for n_idx in np.nonzero(feature)[0]:
                if n_idx not in attrib:
                    attrib[n_idx] = len(attrib)
                x[n_idx] = feature[n_idx]
            y = np.array(list(map(int, label_str.split(","))))
            if not np.any(y):
                continue
            Y.append(y)
            X.append(x)

Y = np.array(Y)
YT= np.transpose(Y)
for idx, y in enumerate(YT):
    if not np.any(y):
        print("idx {} not occur!".format(idx))

X_trn = []
for e_dict in X:
    x = np.zeros(len(attrib))
    for idx, feature in e_dict.items():
        x[attrib[idx]] = feature
    X_trn.append(x)


X = np.array(X_trn)

print(Y)
classif = OneVsRestClassifier(SVC(kernel='linear'))

classif.fit(X,Y)
print("classify train over")
X = []
Y = []

file_name = test_filename
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
        feature = np.array(list(map(float, feature.split(","))))
        x = dict()
        for n_idx in np.nonzero(feature)[0]:
            if n_idx not in attrib:
                attrib[n_idx] = len(attrib)
            x[n_idx] = feature[n_idx]
        y = np.array(list(map(int, label_str.split(","))))
        if not np.any(y): continue
        Y.append(y)
        X.append(x)
X_pred = []
for e_dict in X:
    x = np.zeros(len(attrib))
    for idx, feature in e_dict.items():
        x[attrib[idx]] = feature
    X_pred.append(x)


X = np.array(X_pred)
gt_Y = np.array(Y)


pred_Y = classif.predict(X)
score = f1_score(gt_Y,pred_Y , average='weighted')
print(score)  # this score is for check generate feature is valid feature
