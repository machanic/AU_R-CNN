import numpy as np
import re
import cv2
from config import DATA_PATH,CROP_DATA_PATH,AU_REGION_MASK_PATH
import os

def generate_image():
    video_pattern = re.compile('.*Video(.+?)_.*?.avi', re.DOTALL)
    video_sub_list = ["Video_RightCamera", "Videos_LeftCamera"]
    for video_sub in video_sub_list:
        for video_name in os.listdir(DATA_PATH["DISFA"]+os.sep + video_sub):
            matcher = video_pattern.match(video_name)
            if matcher:
                subject_name = matcher.group(1)
            video_path = DATA_PATH["DISFA"]+os.sep + video_sub + os.sep + video_name
            print(video_path)
            cap = cv2.VideoCapture(video_path)

            while (cap.isOpened()):
                ret, frame = cap.read()
                outpath = CROP_DATA_PATH["DISFA"] + os.sep + video_sub + os.sep + subject_name + os.sep + frame + ".jpg"
                print("write frame:{}".format(outpath))
                cv2.imwrite(outpath, frame)



if __name__ == "__main__":
    generate_image()