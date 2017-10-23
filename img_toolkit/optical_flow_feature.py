# -*- coding: utf-8 -*-
# Created by liangzh0707 on 2017/3/11

import math

import cv2
import numpy as np


class OpticalFlow:
    def __init__(self):
        self.colorwheel = self.makecolorwheel()

    def makecolorwheel(self):
        colorwheel = []
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6
        for i in range(0, RY):
            colorwheel.append((255.0, 255.0 * i / RY, 0))
        for i in range(0, YG):
            colorwheel.append((255.0 - 255.0 * i / YG, 255.0, 0))
        for i in range(0, GC):
            colorwheel.append((0, 255.0, 255.0 * i / GC))
        for i in range(0, CB):
            colorwheel.append((0, 255.0 - 255.0 * i / CB, 255.0))
        for i in range(0, BM):
            colorwheel.append((255.0 * i / BM, 0, 255.0))
        for i in range(0, MR):
            colorwheel.append((255.0, 0, 255.0 - 255.0 * i / MR))
        return np.array(colorwheel, dtype=np.int)

    def visual_flow(self, flow):
        """
        使用木塞尔颜色描述光流形态 将而为通道的光流，转换成三维通道的RGB值
        :param flow:
        :return:
        """
        shape = flow.shape
        color = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        colorwheel = self.colorwheel
        maxrad = -1.0
        UNKNOWN_FLOW_THRESH = 400
        for i in range(shape[0]):
            for j in range(shape[1]):
                flow_at_point = flow[i, j]
                fx = flow_at_point[0]
                fy = flow_at_point[1]
                if np.fabs(fx) > UNKNOWN_FLOW_THRESH or np.fabs(fy) > UNKNOWN_FLOW_THRESH:
                    continue
                rad = np.sqrt(fx * fx + fy * fy)
                if maxrad > rad:
                    maxrad = maxrad
                else:
                    maxrad = rad

        for i in range(shape[0]):
            for j in range(shape[1]):
                flow_at_point = flow[i, j]

                fx = flow_at_point[0] / maxrad
                fy = flow_at_point[1] / maxrad
                if np.fabs(fx) > UNKNOWN_FLOW_THRESH or np.fabs(fy) > UNKNOWN_FLOW_THRESH:
                    color[i, j] = (0, 0, 0)
                    continue
                rad = np.sqrt(fx * fx + fy * fy)
                angle = math.atan2(-fy, -fx) / cv2.cv.CV_PI
                fk = (angle + 1.0) / 2.0 * (len(colorwheel) - 1)
                k0 = int(fk)
                k1 = (k0 + 1) % len(colorwheel)
                f = fk - k0
                for b in range(3):
                    col0 = colorwheel[k0][b] / 255.0
                    col1 = colorwheel[k1][b] / 255.0
                    col = (1 - f) * col0 + f * col1
                    if rad <= 1:
                        col = 1 - rad * (1 - col)
                    else:
                        col *= .75
                    color[i, j][2 - b] = int(255.0 * col)
        return color

    def draw_flow(self, im, flow, step=8):
        """
        在图片上绘制光流箭头
        :param im:
        :param flow:
        :param step:
        :return:
        """
        h, w = im.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
        fx, fy = flow[y, x].T

        # create line endpoints
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        # create image and draw
        vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def calc_flow(self, img1, img2):
        """
        输入两张图片，单通道的灰度图或者三通道的RGB图，返回一个光流
        :param img1:
        :param img2:
        :return:
        """
        gray1 = None
        gray2 = None
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)[0]
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)[0]

        elif len(img1.shape) == 2:
            gray1 = np.copy(img1)
            gray2 = np.copy(img2)
        else:
            assert False, "输入的图像通道数错误"

        # 这个光流方法是opencv自带的使用的是2003年的有点老
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, 0.5, 4, 25, 3, 5, 1.5, 0)
        return flow

    def calc_affine_mat(self, feature_point_front, feature_point_back):
        # 要扩展一维常量1才能方便计算

        X = np.pad(feature_point_front, (0, 1), 'constant', constant_values=(0, 1))[:-1]
        Y = np.pad(feature_point_back, (0, 1), 'constant', constant_values=(0, 1))[:-1]
        # l2范式的解析解, 这部分可能使用l1范式求解
        T = np.dot(np.linalg.inv(np.dot(X.T, X) + 0.000001 * np.eye(3)), np.dot(X.T, Y))

        # R = np.dot(X, T)
        # for i in R:
        #     print i[0] / i[2], i[1] / i[2]
        # print Y

        return T


