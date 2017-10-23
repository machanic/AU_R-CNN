
import cv2
import dlib
import numpy as np
from functools import lru_cache
from img_toolkit.geometry_utils import sort_clockwise
from img_toolkit.face_region_mask import face_img_mask

from design_pattern.decorator import Singleton
from AU_rcnn import transforms
import config

class FaceLandMark(object, metaclass=Singleton):

    def __init__(self, model_file_path):
        self.predictor = dlib.shape_predictor(model_file_path)
        self.detector = dlib.get_frontal_face_detector()
        print("FaceLandMark init call! {}".format(model_file_path))

    def landmark_from_path(self, face_file_path):
        image = cv2.imread(face_file_path,cv2.IMREAD_GRAYSCALE)
        return self.landmark(image)


    def landmark(self, image, need_txt_img=False):
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = image

        if image.ndim >= 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # clahe_image = clahe.apply(gray)
        small_gray_img = cv2.resize(gray, (int(gray.shape[1] * 1/4.0), int(gray.shape[0] * 1/4.0)))

        dets = self.detector(small_gray_img, 0)  # boost speed for small image, detect bounding box of face , slow legacy
        # only one face,so the dets will always be length = 1
        small_d = dets[0]
        # dlib.dlib.rectangle(d.left(),d.top(),d.right(),d.bottom())
        d = dlib.dlib.rectangle(small_d.left() * 4, small_d.top() * 4, small_d.right()* 4, small_d.bottom() * 4)
        shape = self.predictor(gray, d)
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_image = None
        if need_txt_img:
            new_image = image.copy()
            for i in range(1, 68):
                cv2.putText(new_image, str(i), (shape.part(i).x,
                                                shape.part(i).y), font, 0.4, (255, 255, 255), 1)
                cv2.circle(new_image, (shape.part(i).x, shape.part(i).y),
                           1, (0, 0, 255), thickness=1)
        return {i: (shape.part(i).x, shape.part(i).y)
                    for i in range(1, 68)}, image, new_image

    def split_ROI(self, landmark):

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
        for roi_no, landmark_ls in config.ROI_LANDMARK.items():
            polygon_arr = trans_landmark2pointarr(landmark_ls)
            polygon_arr = polygon_arr.astype(np.int32)
            polygons[int(roi_no)] = polygon_arr
        return polygons


if __name__ == "__main__":
    land = FaceLandMark(
        config.DLIB_LANDMARK_PRETRAIN)
    face_img_path = "D:/0233.jpg"
    trn_img = cv2.imread(face_img_path, cv2.IMREAD_COLOR)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #
    # trn_img = clahe.apply(trn_img)
    # trn_img = trn_img[...,np.newaxis]
    #mask = np.zeros(trn_img.shape[:-1], np.uint8)
    landmark, _ , _= land.landmark(image=trn_img)
    roi_polygons = land.split_ROI(landmark)
    in_size = trn_img.shape
    print(trn_img.shape[0], trn_img.shape[0] * 1 / 4.0)
    # trn_img = cv2.resize(trn_img, (round(trn_img.shape[1] * 1/4.0), round(trn_img.shape[0] * 1/4.0)))
    out_size = trn_img.shape
    print(out_size)
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    print(y_scale, x_scale)
    for roi_no, polygon_vertex_arr in roi_polygons.items():
        # if int(roi_no) == 40 or int(roi_no) == 41:
        polygon_vertex_arr[0, :] = np.round(x_scale * polygon_vertex_arr[0, :])
        polygon_vertex_arr[1, :] = np.round(y_scale * polygon_vertex_arr[1, :])
        polygon_vertex_arr= sort_clockwise(polygon_vertex_arr.tolist())
        cv2.polylines(trn_img, [polygon_vertex_arr], True, (2,0, 200), thickness=4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(trn_img, str(roi_no), tuple(np.mean(polygon_vertex_arr,axis=0).astype(np.int32)), font,1,(0,255,255),thickness=2)

    for AU in config.AU_ROI.keys():
        copy_face = trn_img.copy()
        mask = face_img_mask(AU, face_img_path, land)
        color_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        color_mask[mask!=0] = [200,0,100]
        new_face = cv2.add(trn_img, color_mask)

        # cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
        cv2.imwrite("D:\\AU={}.jpg".format(AU), new_face)

        # cv2.resizeWindow('mask', 300, 500)

        # cv2.waitKey(0)

    #mark, newface = land.landmark(face_file_path="/home2/mac/yonna.jpg")
    #cv2.imwrite("/home2/mac/yonna2.jpg", newface)
