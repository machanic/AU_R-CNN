import sys
import os
import dlib
import glob
import cv2
import numpy as np
from geometry_utils import sort_clockwise

from config import ROI_LANDMARK, CV_TRAIN_MODEL


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls._instance = None

    def __call__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kw)
        return cls._instance


class FaceLandMark(object):
    __metaclass__ = Singleton

    def __init__(self, model_file_path):
        self.predictor = dlib.shape_predictor(model_file_path)
        self.detector = dlib.get_frontal_face_detector()
        print("FaceLandMark init call!")

    def landmark(self, image=None, face_file_path=None):
        if image is None:
            image = cv2.imread(face_file_path)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = image
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        dets = self.detector(clahe_image, 1)
        # only one face,so the dets will always be length = 1

        d = dets[0]

        # d.left(), d.top(), d.right(), d.bottom()
        shape = self.predictor(clahe_image, d)
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_image = image.copy()
        for i in range(1, 68):
            cv2.putText(new_image, str(i), (shape.part(i).x,
                                            shape.part(i).y), font, 0.3, (255, 255, 255), 1)
            cv2.circle(new_image, (shape.part(i).x, shape.part(i).y),
                       1, (0, 0, 255), thickness=1)
        return {i: (shape.part(i).x, shape.part(i).y)
                for i in range(1, 68)}, new_image

    def split_ROI(self, image, landmark):

        def trans_landmark2pointarr(landmark_ls):
            point_arr = []
            for land in landmark_ls:
                if land.endswith("uu"):
                    land = int(land[:-2])
                    x, y = landmark[land]
                    y -= 40
                    point_arr.append((x, y))
                elif land.endswith("u"):
                    land = int(land[:-1])
                    x, y = landmark[land]
                    y -= 20
                    point_arr.append((x, y))
                elif "~" in land:
                    land_a, land_b = land.split("~")
                    land_a = int(land_a)
                    land_b = int(land_b)
                    x = (landmark[land_a][0] + landmark[land_b][0]) / 2
                    y = (landmark[land_a][1] + landmark[land_b][1]) / 2
                    point_arr.append((x, y))
                else:
                    x, y = landmark[int(land)]
                    point_arr.append((x, y))
            return sort_clockwise(point_arr)

        polygons = {}
        for roi_no, landmark_ls in ROI_LANDMARK.items():
            polygon_arr = trans_landmark2pointarr(landmark_ls)
            polygon_arr = polygon_arr.astype(np.int32)
            polygons[int(roi_no)] = polygon_arr
        return polygons
        #cv2.polylines(image, polygons, True, (0,255,255))
        # cv2.imshow("ROI",image)
        # cv2.waitKey(0)


if __name__ == "__main__":
    land = FaceLandMark(
        CV_TRAIN_MODEL +
        os.sep +
        "shape_predictor_68_face_landmarks.dat")
    mark, newface = land.landmark(face_file_path="/home2/mac/yonna.jpg")
    cv2.imwrite("/home2/mac/yonna2.jpg", newface)
