import cv2
import numpy as np
import img_toolkits.optical_flow_feature as of
from feature_extract.MDMO import MDMOFeature


fl = of.OpticalFlow()


def face_alignment_flow(img1, img2, flow_path):
    flow_result = np.load(flow_path)
    landmark, new_face = MDMOFeature.land.landmark(image=img1)

    contour_landmark = [i for i in range(1, 16)] + [30]
    # feature_pos each is (x,y)
    feature_pos = [flow_result[no] for no in contour_landmark]
    feature_back = []
    for pos in feature_pos:
        new_pos = (pos[0] + flow_result[pos[1], pos[0]][0],
                   pos[1] + flow_result[pos[1], pos[0]][1])
        feature_back.append(new_pos)
    convert_mat = fl.calc_affine_mat(
        np.array(
            feature_back, dtype=np.float32), np.array(
            feature_pos, dtype=np.float32))
    B = cv2.warpPerspective(im2, convert_mat.T, (img1.shape[1], img2.shape[0]))
    return B
