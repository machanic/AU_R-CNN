import cv2
import numpy as np

class RoISift(object):

    def extract(self, image, bbox, layer=None):
        # image shape = (channel, height,width)
        image = np.transpose(image, (1,2,0))  # shape convert to (height, width, channel)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        roi_features = []
        for box in bbox:
            mask = np.zeros_like(gray)
            y_min, x_min, y_max, x_max = box
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1, -1)
            masked_data = gray * mask
            kp, des = sift.detectAndCompute(masked_data, None)
            roi_feature = np.sum(des, axis=0)  # sum all point vector inside each roi
            roi_features.append(roi_feature)
        return np.stack(roi_features)  # shape = R x 128
