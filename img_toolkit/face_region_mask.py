import config
import numpy as np
import cv2
import warnings


def calculate_offset_polygon_arr(rect, new_face_image, polygon_arr):
    top, left, width, height = rect["top"], rect["left"], rect["width"], rect["height"]
    polygon_arr = polygon_arr.astype(np.float32)
    polygon_arr -= np.array([left, top])
    polygon_arr *= np.array([float(new_face_image.shape[1])/ width,
                             float(new_face_image.shape[0])/ height])
    polygon_arr = polygon_arr.astype(np.int32)
    return polygon_arr



path_landmark_dict = {}
def face_img_mask(action_unit_no, face_file_path, landmarker):
    if face_file_path in path_landmark_dict:
        landmark = path_landmark_dict[face_file_path]
        image = cv2.imread(face_file_path)
    else:
        landmark, image, _ = landmarker.landmark(face_file_path=face_file_path)
        path_landmark_dict[face_file_path] = landmark
    mask = np.zeros(image.shape[:2], np.uint8)  # note that channel is LAST axis

    roi_polygons = landmarker.split_ROI(landmark)
    region_lst = config.AU_ROI[str(action_unit_no)]
    for roi_no, polygon_vertex_arr in roi_polygons.items():
        if roi_no in region_lst:
            cv2.fillConvexPoly(mask, polygon_vertex_arr, 1)
    return mask


# this method is far more speed. use it
def crop_face_mask_from_landmark(action_unit_no, landmark, new_face_image, rect_dict, landmarker):
    mask = np.zeros(new_face_image.shape[:2], np.uint8)  # note that channel is LAST axis
    roi_polygons = landmarker.split_ROI(landmark)
    region_lst = config.AU_ROI[str(action_unit_no)]
    for roi_no, polygon_vertex_arr in roi_polygons.items():
        if roi_no in region_lst:
            polygon_vertex_arr = calculate_offset_polygon_arr(rect_dict, new_face_image, polygon_vertex_arr)
            cv2.fillConvexPoly(mask, polygon_vertex_arr, 50)
    return mask


def crop_face_img_mask(action_unit_no, orig_face_image_path, new_face_image, rect_dict, landmarker):
    '''
    crop from original face image file path, landmark also from original face image
    mask is obtained from original face image, then to offset it to adaptive new cropped face.
    :param action_unit_no: actual AU
    :param orig_image: obtained by cv2.imread
    :param rect_dict: {"top":xxx, "left":xxx, "width":xxx, "height":xxx }
    :param landmarker: landmarker
    :return: mask: np.ndarray: shape=H x W
    '''
    warnings.warn(
        'crop_face_img_mask is deprecated. Use crop_face_mask_from_landmark instead.',
        DeprecationWarning)
    landmark, _, _ = landmarker.landmark_from_path(orig_face_image_path)
    return  crop_face_mask_from_landmark(action_unit_no,landmark,new_face_image, rect_dict, landmarker)