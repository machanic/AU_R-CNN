# -*- coding: utf-8 -*-
# Created by liangzh0707 on 2017/3/1

import cv2
import numpy as np

import os
if __name__ == '__main__':
    image_path = "../dataset/001/S005_001_00000001.png"
    print os.path.abspath(image_path)
    assert os.path.exists(image_path)

    img = np.zeros((512,512))
    cv2.fillConvexPoly(img,np.array([(250,250),(250,270),(290,270),(290,250)]),(255))

    w, h = img.shape
    mat = np.float32(img) / 255.0 / w / h

    cv2.imshow("a", mat)
    cv2.waitKey(0)
    feq = cv2.dft(mat, None, cv2.DFT_COMPLEX_OUTPUT)
    print feq.shape
    a = feq[:, :, 0]
    b = feq[:, :, 1]

    f = np.power(np.add(a * a, b * b), 0.5)
    output = cv2.normalize(np.log(f+0.00000001), None, 0, 1, cv2.NORM_MINMAX)

    cv2.imshow("a", output)
    cv2.waitKey(0)

    pass
