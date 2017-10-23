from sklearn.preprocessing import MultiLabelBinarizer
import os
import skmultilearn
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss


def load_data(folder_dir):
    labels = []
    features = []
    for file_name in os.listdir(folder_dir):
        if not file_name.endswith(".txt"):
            continue
        with open(folder_dir + os.sep + file_name, "r") as file_obj:
            for line in file_obj:
                if line.startswith("#edge"):
                    continue
                line = line.strip()
                lines = line.split()
                label = lines[1][1:-1]
                label = list(map(int, label.split(",")))
                all_zero = all(v == 0 for v in label)
                if all_zero:
                    continue
                labels.append(label)
                feature = lines[2]
                feature = feature[len("features:"):]
                features.append(list(map(float, feature.split(","))))
    print("load over")
    labels = np.asarray(labels, np.int32)
    features = csr_matrix(np.array(features), dtype=np.float32)
    return features, labels

def train(X, y):
    classifier = BinaryRelevance(classifier = SVC(), require_dense = [False, True])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print("before train")
    classifier.fit(X_train, y_train)
    print("train over begin predict")
    predictions = classifier.predict(X_test)
    print("validate loss accuracy: {}".format(1 - hamming_loss(y_test, np.stack(predictions))))


if __name__ == "__main__":
    folder_path = "/home/machen/face_expr/result/graph_backup/3_3"
    X, y = load_data(folder_path)
    train(X, y)