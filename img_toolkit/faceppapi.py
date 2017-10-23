# coding:utf-8

import json

import requests
import cv2
from img_toolkit.face_crop import dlib_face_rect

def get_feature_point(image_file, verbose=False):
    """
    通过人脸图像获取举行位置和全部83个特征点
    :param image:  通过'rb'模式打开的二进制图片文件
    :return:
    """

    url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    param = {
        "api_key": "H8qZAoXMbP4X96zmoP35WHF4ZX_d4iLN",
        "api_secret": "VC2ztQ4ME5SnVDwOWO4s-uBQdH1r83Cp",
        "return_landmark": 1
    }
    files = {'image_file': ('image_file', image_file, 'multipart/form-data')}
    response = requests.post(url, data=param, files=files)

    if verbose:
        print(response.content)

    jd = json.JSONDecoder()
    return jd.decode(response.content)


def get_feature_points_fromimage(image, verbose=False):
    """
    通过图片获取特征点，通过人脸图像获取举行位置和全部83个特征点，先保存成文件，然后再上传这个文件
    :param image:  numpy格式的图片
    :return:
    """
    cv2.imwrite("file_tmp.jpg", image)
    img_file = open("file_tmp.jpg", "rb")

    url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    param = {
        "api_key": "H8qZAoXMbP4X96zmoP35WHF4ZX_d4iLN",
        "api_secret": "VC2ztQ4ME5SnVDwOWO4s-uBQdH1r83Cp",
        "return_landmark": 1
    }
    files = {'image_file': ('image_file', img_file, 'multipart/form-data')}
    response = requests.post(url, data=param, files=files,timeout=3)

    if verbose:
        print(response.content)

    jd = json.JSONDecoder()
    if len(jd.decode(response.content)['faces']) == 0:
        return None
    else:
        return jd.decode(response.content)['faces'][0]['landmark']



def get_face_rect(image, verbose=False):
    """
    检测人脸所在的矩形框
    :param image:
    :param verbose:
    :return:
    """
    url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    param = {
        "api_key": "H8qZAoXMbP4X96zmoP35WHF4ZX_d4iLN",
        "api_secret": "VC2ztQ4ME5SnVDwOWO4s-uBQdH1r83Cp",
        "return_landmark": 0
    }
    files = {'image_file': ('image_file', image, 'multipart/form-data')}
    response = requests.post(url, data=param, files=files, timeout=10)

    if verbose:
        print(response.content)

    jd = json.JSONDecoder()
    if len(jd.decode(response.content)['faces']) == 0:
        return None
    else:
        return jd.decode(response.content)['faces'][0]['face_rectangle']



def from_filename2face(file_name, face_size, verbose=False):
    file_src = open(file_name, "rb")
    image = cv2.imread(file_name)
    rect = dlib_face_rect(image, verbose=verbose)
    file_src.close()
    if rect is None:
        return None
    else:
        face = image[rect['top']-20: rect['top'] + rect['height'], rect['left']:rect['left'] + rect['width'], :]
        face = cv2.resize(face, face_size)
    return face

def from_face2file(file_in, file_out, face_size, verbose=False):
    face = from_filename2face(file_in, face_size,verbose)
    cv2.imwrite(file_out, face)



face_cascade = None
def opencv_from_face2file(file_in, file_out, face_size, verbose=False):
    global face_cascade
    if face_cascade == None:
        face_cascade = cv2.CascadeClassifier(r'{}/haarcascade_frontalface_default.xml'.format(CV_PRETRAIN_MODEL))
    face_raw = cv2.imread(file_in)
    faces = face_cascade.detectMultiScale(face_raw,
    scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (5,5),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    face = faces[0]
    left = face[0]
    top = face[1]
    width = face[2]
    height = face[3]
    cropped = face_raw[top: top + height, left: left + width,:]
    cropped = cv2.resize(cropped, face_size)
    cv2.imwrite(file_out, cropped)

