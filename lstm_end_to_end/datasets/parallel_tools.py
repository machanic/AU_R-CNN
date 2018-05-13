from img_toolkit.face_mask_cropper import FaceMaskCropper, crop_face_mask_from_landmark
from collections import defaultdict
import numpy as np
import config
import cv2
import dill

def apply_packed_function_for_map(target_function, item, kwargs):
    """
    Unpack dumped function as target function and call it with arguments.

    :param (dumped_function, item, args, kwargs):
        a tuple of dumped function and its arguments
    :return:
        result of target function
    """
    try:
        res = target_function(*item, **kwargs)
        res = dill.dumps(res)
    except IndexError:
        return None, None, None
    return res


def pack_function_for_map(target_function, items, **kwargs):
    """
    Pack function and arguments to object that can be sent from one
    multiprocessing.Process to another. The main problem is:
        «multiprocessing.Pool.map*» or «apply*»
        cannot use class methods or closures.
    It solves this problem with «dill».
    It works with target function as argument, dumps it («with dill»)
    and returns dumped function with arguments of target function.
    For more performance we dump only target function itself
    and don't dump its arguments.
    How to use (pseudo-code):

        ~>>> import multiprocessing
        ~>>> images = [...]
        ~>>> pool = multiprocessing.Pool(100500)
        ~>>> features = pool.map(
        ~...     *pack_function_for_map(
        ~...         super(Extractor, self).extract_features,
        ~...         images,
        ~...         type='png'
        ~...         **options,
        ~...     )
        ~... )
        ~>>>

    :param target_function:
        function, that you want to execute like  target_function(item, *args, **kwargs).
    :param items:
        list of items for map
    :param args:
        positional arguments for target_function(item, *args, **kwargs)
    :param kwargs:
        named arguments for target_function(item, *args, **kwargs)
    :return: tuple(function_wrapper, dumped_items)
        It returs a tuple with
            * function wrapper, that unpack and call target function;
            * list of packed target function and its' arguments.
    """
    dumped_items = [(target_function, item, kwargs) for item in items]
    return apply_packed_function_for_map, dumped_items

def parallel_landmark_and_conn_component(img_path, landmark_dict, AU_box_dict):

    orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if landmark_dict is None or len(landmark_dict) == 0:
        try:
            landmark_dict, _, _ = FaceMaskCropper.landmark.landmark(image=orig_img, need_txt_img=False) # slow
        except IndexError:
            if AU_box_dict is None:
                AU_box_dict = defaultdict(list)
                for AU in config.AU_ROI.keys():
                    if AU in config.SYMMETRIC_AU:
                        for _ in range(2):
                            AU_box_dict[AU].append((0.0, 0.0, config.IMG_SIZE[1], config.IMG_SIZE[0]))
                    else:
                        AU_box_dict[AU].append((0.0, 0.0, config.IMG_SIZE[1], config.IMG_SIZE[0]))
            return img_path, AU_box_dict, None, True

    cropped_face, rect = FaceMaskCropper.dlib_face_crop(orig_img, landmark_dict)
    cropped_face = cv2.resize(cropped_face, config.IMG_SIZE)
    del orig_img
    if AU_box_dict is None:
        AU_box_dict = defaultdict(list)
        for AU in config.AU_ROI.keys():
            mask = crop_face_mask_from_landmark(AU, landmark_dict, cropped_face, rect, landmarker=FaceMaskCropper.landmark)
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
    for AU, box_lst in AU_box_dict.items():
        AU_box_dict[AU] = sorted(box_lst, key=lambda e: int(e[3]))
    return img_path, AU_box_dict, landmark_dict, False
