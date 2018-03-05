import cv2
import functools
from img_toolkit.face_landmark import FaceLandMark
import config
from img_toolkit.face_region_mask import crop_face_mask_from_landmark
import numpy as np
from collections import defaultdict


class FaceMaskCropper(object):
    landmark = FaceLandMark(config.DLIB_LANDMARK_PRETRAIN)

    @staticmethod
    def _dlib_face_rect(image, landmark_dict):
        h_offset = 50
        w_offset = 20
        sorted_x = sorted([val[0] for val in landmark_dict.values()])
        sorted_y = sorted([val[1] for val in landmark_dict.values()])
        rect = {"top": sorted_y[0] - h_offset, "left": sorted_x[0] - w_offset,
                "width": sorted_x[-1] - sorted_x[0] + 2 * w_offset, "height": sorted_y[-1] - sorted_y[0] + h_offset}
        for key, val in rect.items():
            if val < 0:
                rect[key] = 0
        new_face = image[rect["top"]:rect["top"] + rect["height"], rect["left"]: rect["left"] + rect["width"], ...]
        return new_face, rect

    @staticmethod
    def get_cropface_and_mask(orig_img_path, channel_first=True, mc_manager=None):
        if mc_manager is not None:
            try:
                if orig_img_path in mc_manager:
                    result = mc_manager.get_crop_mask(orig_img_path)
                    return result
            except Exception:
                pass

        orig_img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
        if orig_img is None:
            print("error read image: {}".format(orig_img_path))
        landmark_dict, _, _ = FaceMaskCropper.landmark.landmark(image=orig_img, need_txt_img=False)
        assert len(landmark_dict) == 67
        new_face, rect = FaceMaskCropper._dlib_face_rect(orig_img, landmark_dict)
        del orig_img
        new_face = cv2.resize(new_face, config.IMG_SIZE)
        AU_mask_dict = dict()
        for AU in config.AU_ROI.keys():
            mask = crop_face_mask_from_landmark(AU, landmark_dict, new_face, rect, FaceMaskCropper.landmark)
            AU_mask_dict[AU] = mask
        if channel_first:
            new_face = np.transpose(new_face, (2,0,1))
        if mc_manager is not None:
            try:
                mc_manager.set_crop_mask(orig_img_path, new_face, AU_mask_dict)
            except Exception:
                pass
        return new_face, AU_mask_dict


    @staticmethod
    def get_cropface_and_box(orig_img_path, channel_first=True, mc_manager=None, key_prefix=""):
        key = key_prefix+orig_img_path
        if mc_manager is not None:
            try:
                if key in mc_manager:
                    orig_img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
                    result = mc_manager.get(key)
                    crop_rect = result["crop_rect"]
                    AU_box_dict = result["AU_box_dict"]
                    new_face = orig_img[crop_rect["top"]:crop_rect["top"] + crop_rect["height"],
                               crop_rect["left"]: crop_rect["left"] + crop_rect["width"], ...]
                    new_face = cv2.resize(new_face, config.IMG_SIZE)
                    if channel_first:
                        new_face = np.transpose(new_face, (2, 0, 1))
                    return new_face, AU_box_dict
            except Exception:
                pass
        orig_img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
        landmark_dict, _, _ = FaceMaskCropper.landmark.landmark(orig_img, need_txt_img=False)
        new_face, rect = FaceMaskCropper._dlib_face_rect(orig_img, landmark_dict)
        new_face = cv2.resize(new_face, config.IMG_SIZE)

        del orig_img
        AU_box_dict =defaultdict(list)

        for AU in config.AU_ROI.keys():
            mask = crop_face_mask_from_landmark(AU, landmark_dict, new_face, rect, landmarker=FaceMaskCropper.landmark)
            connect_arr = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)  # mask shape = 1 x H x W
            component_num = connect_arr[0]
            label_matrix = connect_arr[1]
            # convert mask polygon to rectangle
            for component_label in range(1, component_num):

                row_col = list(zip(*np.where(label_matrix == component_label)))
                row_col = np.array(row_col)
                y_min_index = np.argmin(row_col[:, 0])
                y_min = row_col[y_min_index, 0]
                x_min_index = np.argmin(row_col[:, 1])
                x_min = row_col[x_min_index, 1]
                y_max_index = np.argmax(row_col[:, 0])
                y_max = row_col[y_max_index, 0]
                x_max_index = np.argmax(row_col[:, 1])
                x_max = row_col[x_max_index, 1]
                # same region may be shared by different AU, we must deal with it
                coordinates = (y_min, x_min, y_max, x_max)

                if y_min == y_max and x_min == x_max:  # 尖角处会产生孤立的单个点，会不会有一个mask只有尖角？
                    # print(("single point mask: img:{0} mask:{1}".format(self._images[i], mask_path)))
                    # 然后用concat_example来拼接起来
                    continue
                AU_box_dict[AU].append(coordinates)
            del label_matrix
            del mask
        if mc_manager is not None:
            try:
                save_dict = {"crop_rect":rect, "AU_box_dict":AU_box_dict}
                mc_manager.set(key, save_dict)
            except Exception:
                pass
        if channel_first:
            new_face = np.transpose(new_face, (2, 0, 1))
        for AU, box_lst in AU_box_dict.items():
            AU_box_dict[AU] = sorted(box_lst, key=lambda e:int(e[3]))
        return new_face, AU_box_dict



if __name__ == "__main__":

    new_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask("/home/machen/dataset//BP4D/BP4D-training//M018/T8/187.jpg", channel_first=False)
    cv2.imwrite("/home/machen/tmp/newface.jpg", new_face)
    for AU, mask in AU_mask_dict.items():
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        new_mask = cv2.add(new_face, mask)
        cv2.imwrite("/home/machen/tmp/mask_{}.jpg".format(AU), new_mask)