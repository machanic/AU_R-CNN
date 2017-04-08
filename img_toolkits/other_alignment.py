import cv2
import numpy as np
import img_toolkits.optical_flow_feature as of
from feature_extract.MDMO import MDMOFeature
import os

fl = of.OpticalFlow()


def face_alignment_flow(img1, img2, flow_path):
    flow_result = np.load(flow_path)
    landmark, new_face = MDMOFeature.land.landmark(image=img1)

    # print img1.shapes
    # print(landmark)
    contour_landmark = [i for i in range(1, 16)] + [30]
    # feature_pos each is (x,y)
    feature_pos = [list(landmark[no]) for no in contour_landmark]
    # print(feature_pos)
    feature_back = []
    for pos in feature_pos:
        pos[0] = max(pos[0], 0)
        pos[1] = max(pos[1], 0)
        pos[0] = min(pos[0], img2.shape[1] - 1)
        pos[1] = min(pos[1], img2.shape[0] - 1)
        new_pos = (pos[0] + flow_result[pos[1], pos[0]][0],
                   pos[1] + flow_result[pos[1], pos[0]][1])
        feature_back.append(new_pos)
    convert_mat = fl.calc_affine_mat(
        np.array(
            feature_back, dtype=np.float32), np.array(
            feature_pos, dtype=np.float32))
    # print(convert_mat)
    B = cv2.warpPerspective(
        img2, convert_mat.T, (img2.shape[1], img2.shape[0]))
    return B


def align_folder_firstframe(img_folder_path_ls, flow_foler_path_ls):
    newimg_ls = []
    img1_path = img_folder_path_ls[0]
    print("img1 is : " + img1_path)
    for idx, img_path in enumerate(
            img_folder_path_ls[1:]):  # idx start from 1, which indicate 2nd picture
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img_path)
        newimg = face_alignment_flow(img1, img2, flow_foler_path_ls[idx])
        newimg_ls.append(newimg)
    return newimg_ls


def align_folder_firstframe_out(
        img_folder_path_ls,
        flow_folder_path_ls,
        out_folder_path):
    newimg_ls = align_folder_firstframe(
        img_folder_path_ls, flow_folder_path_ls)
    for idx, newimg in enumerate(newimg_ls):
        cv2.imwrite(os.sep.join(
            (out_folder_path, os.path.basename(img_folder_path_ls[idx + 1]))), newimg)
