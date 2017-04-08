from log_utils import LogUtils
from collections import OrderedDict
import os

logger = LogUtils(os.path.expanduser("~/log.txt"))


DATA_PATH = {
    "ck+":"data/CK+",
    "fer2013":"data/fer2013",
    "CAME2": "data/CAME2"
}

'''
READER_CREATER's key must equally match DATA_PATH's key
'''
READER_CREATER = {
    "ck+": "CKPlusDataBuilder",
    "fer2013" : "Fer2013DataBuilder",
    "CAME2" : "CAME2DataBuilder"
}

EMOTION_LABEL_CK = {
    0: "neutral",
    1: "anger",
    2: "contempt",
    3: "disgust",
    4: "fear",
    5: "happiness",
    6: "sadness",
    7: "surprise",
}

EMOTION_LABEL_CASME2 = {
    "happiness": 0,
    "disgust": 1,
    "repression": 2,
    "surprise": 3,
    "fear": 4,
    "sadness": 5,
    "others": 6
}

IMG_SIZE = (227,227)
MEAN_VALUE = 128
TRN_TEST_FOLD = 5

CV_TRAIN_MODEL = "data/cv_train_model"
PY_CAFFE_PATH = "/home2/mac/caffe_orig/python"
CAFFE_PATH = "/home2/mac/caffe_mac/build/tools/caffe"

BACKEND = "lmdb"
BACKEND_DIR = "CAFFE_IN"

#key is ROI number
ROI_LANDMARK = OrderedDict({"1": ["17u","19u","19","17"], #eye brow
                "2": ["19u","21u","21","19"],
                "3": ["21u", "27uu","27","21"],
                "4": ["27uu","22u","27","22"],
                "5": ["22u","24u","24","22"],
                "6": ["24u","26u","26","24"],
                #eye and temple
                "7": ["1", "17", "36"],
                "8": ["17","19","37","36"],
                "9": ["19","38","39","21"],
                "10":["21","27","28","39"],
                "11":["22","27","28", "42"],
                "12":["22","42","43","24"],
                "13":["24","44","45","26"],
                "14":["26","16","15","45"],
                #middle
                "15":["2","41~2","3~29","3"],
                "16":["41","2~41","3~29","39"],
                "17":["39","3~29","29","28"],
                "18":["28","29","13~29","42"],
                "19":["42","46","14~46","13~29"],
                "20":["14~46","14","13","13~29"],
                #middle down
                "21":["3","3~29","4~33","4"],
                "22":["4~33","3~29","29","33"],
                "23":["33","29","13~29","12~33"],
                "24":["13~29","13","12","12~33"],
                #mouse
                "25":["4","4~33","5~59","5"],
                "26":["5~59","4~33","33","59"],
                "27":["33","55","11~55","12~33"],
                "28":["11~55","12~33","12","11"],
                #below mouse
                "29":["59","58","6~58","5~59"],
                "30":["6~58","58","57","8~57"],
                "31":["57","8~57","10~56","56"],
                "32":["56","10~56","11~55","55"],
                #chin
                "33":["5","5~59","6~58","6"],
                "34":["6","6~58","8~57","8"],
                "35":["8~57","10~56","10","8"],
                "36":["10~56","10","11","11~55"]
                })