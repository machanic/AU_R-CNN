
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from operator import itemgetter
import os

from config import ROI_LANDMARK, CV_TRAIN_MODEL
from face_landmark import FaceLandMark


class MDMOFeature(object):

    land = FaceLandMark(
        CV_TRAIN_MODEL +
        os.sep +
        "shape_predictor_68_face_landmarks.dat")

    def __init__(self, face_imagepath_seq, flow_mat_path_ls):
        assert len(face_imagepath_seq) == len(flow_mat_path_ls)
        self.roi_dict_ls = []
        for face_image_path in face_imagepath_seq:
            face_image = cv2.imread(face_image_path)
            landmark, new_face = MDMOFeature.land.landmark(image=face_image)
            roi_polygons = MDMOFeature.land.split_ROI(face_image, landmark)
            self.roi_dict_ls.append(roi_polygons)
        self.flow_mat_sequence = [
            np.load(flow_mat_path) for flow_mat_path in flow_mat_path_ls]
        assert len(self.roi_dict_ls) == len(flow_mat_path_ls)

    # extract 36 x 2 = 72 feature
    def extract_img(self, roi_dict, flow_mat):

        mask = np.zeros(flow_mat.shape[:-1], dtype=np.uint8)
        roi_pixel = {}  # roi_no => pixel coordinate

        for roi_no, polygon_vertex_arr in roi_dict.items():
            cv2.fillConvexPoly(mask, polygon_vertex_arr, roi_no)
        coordinates = np.nonzero(mask)
        roi_coordinates = defaultdict(list)
        for x, y in zip(
                *coordinates):  # iterater over human face each ROI region
            roi_no = mask[x, y]
            # to fetch flow_mat(delta_x, delta_y)'s angle, convert to polar
            # coordinate
            theta = math.atan2(flow_mat[x, y][1],
                               flow_mat[x, y][0]) * 180 / math.pi
            magnitude = math.sqrt(
                flow_mat[x, y][0] ** 2 + flow_mat[x, y][1] ** 2)
            if theta < 0:
                theta += 360
            roi_coordinates[roi_no].append((magnitude, theta))

        psi_i_frame = []
        for roi_no, theta_mag_ls in roi_coordinates.items():
            theta_ls = map(itemgetter(1), theta_mag_ls)
            hist, bin_edges = np.histogram(
                theta_ls, bins=np.linspace(
                    0, 360, 9).tolist())

            max_bin_idx = np.argmax(hist)
            max_bin_count = hist[max_bin_idx]
            # Return the indices of the bins to which each value in input array
            # belongs.
            bin_idxs = np.digitize(theta_ls, bins=np.linspace(0, 360, 9))

            rho = 0.0
            theta = 0.0
            for bin_idx in bin_idxs:
                if bin_idx == max_bin_idx:
                    _mag, _theta = theta_mag_ls[bin_idx]
                    rho += _mag
                    theta += _theta
            each_u_ik = (rho / max_bin_count, theta / max_bin_count)
            psi_i_frame.append(each_u_ik)
        return psi_i_frame

    def extract(self):

        cartesian_psi = np.zeros((len(ROI_LANDMARK), 2), dtype=float)
        for idx, roi_dict in enumerate(self.roi_dict_ls):  # for each frame
            flow_mat = self.flow_mat_sequence[idx]
            psi_i_frame = self.extract_img(roi_dict, flow_mat)
            # convert to Cartesian coordinate for the reason better explanatory

            for idx, u_ik in enumerate(psi_i_frame):
                x_ik, y_ik = math.cos(
                    u_ik[1] * math.pi / 180.0) * u_ik[0], math.sin(u_ik[1] * math.pi / 180.0) * u_ik[0]
                cartesian_psi[idx][0] += x_ik
                cartesian_psi[idx][1] += y_ik
        # len(self.roi_dict_ls) = face images's count
        cartesian_psi_bar = cartesian_psi / len(self.roi_dict_ls)
        psi_bar = np.zeros_like(cartesian_psi_bar)

        assert psi_bar.ndim == 2
        max_rho = 0
        for i, x_y in enumerate(cartesian_psi_bar):
            x, y = x_y[0], x_y[1]
            psi_bar[i, 0] = math.sqrt(x ** 2 + y ** 2)
            psi_bar[i, 1] = math.atan2(y, x)
            if psi_bar[i, 0] > max_rho:
                max_rho = psi_bar[i, 0]
        for i, rho_theta in enumerate(psi_bar):
            psi_bar[i, 0] = rho_theta[0] / max_rho
        psi_bar = psi_bar.flatten()
        return psi_bar
