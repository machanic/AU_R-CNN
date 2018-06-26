import argparse
import random

from img_toolkit.face_landmark import FaceLandMark
import cv2
import numpy as np
import os
from dataset_toolkit.adaptive_AU_config import adaptive_AU_database
from img_toolkit.geometry_utils import sort_clockwise
import config
from img_toolkit.face_mask_cropper import FaceMaskCropper
from AU_rcnn.links.model.faster_rcnn import FasterRCNNResnet101
from AU_rcnn.links.model.faster_rcnn.faster_rcnn_vgg import FasterRCNNVGG16
from dataset_toolkit.compress_utils import get_zip_ROI_AU, get_AU_couple_child
import chainer
from chainer.dataset.convert import concat_examples
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import chainer.functions as F


class HeatMapGenerator(chainer.Chain):
    def __init__(self, weight_param, use_relu):

        first_fc_weight = weight_param['head/fc/W']  # shape = (1000, 2048)
        first_fc_bias = weight_param['head/fc/b']  # shape = (1000, )
        self.use_relu = use_relu
        second_score_weight = weight_param['head/score/W']  # (22,1000)
        second_score_bias = weight_param['head/score/b']  # (22, )
        super(HeatMapGenerator, self).__init__()
        with self.init_scope():
            self.last_conv = chainer.links.Convolution2D(2048, 1000, 1, nobias=False)
            self.last_conv.W.data = np.expand_dims(np.expand_dims(first_fc_weight, axis=-1), axis=-1)  # shape = (1000, 2048, 1, 1)
            self.last_conv.b.data = first_fc_bias # (1000,)

            self.score_conv = chainer.links.Convolution2D(1000, 22, 1, nobias=False)
            self.score_conv.W.data = np.expand_dims(np.expand_dims(second_score_weight, axis=-1), axis=-1)  # shape=(22,1000,1,1)
            self.score_conv.b.data = second_score_bias  # (22, )

    def generate_AUCouple_ROI_mask_image(self, database_name, img_path, roi_activate):
        adaptive_AU_database(database_name)

        cropped_face, AU_mask_dict = FaceMaskCropper.get_cropface_and_mask(img_path, channel_first=False)
        AU_couple_dict = get_zip_ROI_AU()

        land = FaceLandMark(
            config.DLIB_LANDMARK_PRETRAIN)
        landmark, _, _ = land.landmark(image=cropped_face)
        roi_polygons = land.split_ROI(landmark)
        for roi_no, polygon_vertex_arr in roi_polygons.items():
            polygon_vertex_arr[0, :] = np.round(polygon_vertex_arr[0, :])
            polygon_vertex_arr[1, :] = np.round(polygon_vertex_arr[1, :])
            polygon_vertex_arr = sort_clockwise(polygon_vertex_arr.tolist())
            cv2.polylines(cropped_face, [polygon_vertex_arr], True, (0,0,255), thickness=1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cropped_face, str(roi_no), tuple(np.mean(polygon_vertex_arr, axis=0).astype(np.int32)),
                        font, 0.7, (0, 255, 255), thickness=1)
        already_fill_AU = set()
        AUCouple_face_dict = dict()
        for AU in config.AU_ROI.keys():
            AU_couple = AU_couple_dict[AU]
            if AU_couple in already_fill_AU or AU_couple not in roi_activate:
                continue
            already_fill_AU.add(AU_couple)
            mask = AU_mask_dict[AU]
            color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            color_mask[mask != 0] = (199,21,133)
            new_face = cv2.add(cropped_face, color_mask)
            AUCouple_face_dict[AU_couple] = new_face

        return AUCouple_face_dict



    # generate one class's activate map, weight_for_class
    def generate_activate_roi_map(self, roi_feature_map, roi_orig_size):
        # roi_feature_map = 7 x 7 x 2048
        assert roi_feature_map.shape[0] == 1
        roi_feature_map = self.last_conv(roi_feature_map) # N, 1000, 7, 7
        if self.use_relu:
            roi_feature_map = F.relu(roi_feature_map)  # N,1000, 7, 7
        roi_feature_map = self.score_conv(roi_feature_map) # N, 22, 7, 7  # 22 means class number
        roi_feature_map = F.resize_images(roi_feature_map, roi_orig_size)  # reshape to roi_orig_size

        activation_map = chainer.cuda.to_cpu(roi_feature_map.data)
        activation_map -= activation_map.min()
        activation_map /= activation_map.max()
        activation_map = activation_map[0]  # 22, roi_h, roi_w
        return activation_map



def main():
    parser = argparse.ArgumentParser(
        description='generate Graph desc file script')
    parser.add_argument('--mean', default=config.ROOT_PATH + "BP4D/idx/mean_rgb.npy",
                        help='image mean .npy file')
    parser.add_argument("--image", default='C:/Users/machen/Downloads/tmp/face.jpg')
    parser.add_argument("--model",
                        default="C:/Users/machen/Downloads/tmp/BP4D_3_fold_1.npz")
    parser.add_argument("--pretrained_model_name", '-premodel', default='resnet101')
    parser.add_argument('--database', default='BP4D',
                        help='Output directory')
    parser.add_argument('--device', default=0, type=int,
                        help='GPU device number')
    args = parser.parse_args()
    adaptive_AU_database(args.database)

    if args.pretrained_model_name == "resnet101":
        faster_rcnn = FasterRCNNResnet101(n_fg_class=len(config.AU_SQUEEZE),
                                          pretrained_model="resnet101",
                                          mean_file=args.mean, use_lstm=False,
                                          extract_len=1000)  # 可改为/home/machen/face_expr/result/snapshot_model.npz
    elif args.pretrained_model_name == "vgg":
        faster_rcnn = FasterRCNNVGG16(n_fg_class=len(config.AU_SQUEEZE),
                                      pretrained_model="imagenet",
                                      mean_file=args.mean,
                                      use_lstm=False,
                                      extract_len=1000)

    if os.path.exists(args.model):
        print("loading pretrained snapshot:{}".format(args.model))
        chainer.serializers.load_npz(args.model, faster_rcnn)
    if args.device >= 0:
        faster_rcnn.to_gpu(args.device)
        chainer.cuda.get_device_from_id(int(args.device)).use()

    heatmap_gen = HeatMapGenerator(np.load(args.model), use_relu=True)
    if args.device >= 0:
        heatmap_gen.to_gpu(args.device)
    cropped_face, AU_box_dict = FaceMaskCropper.get_cropface_and_box(args.image,args.image, channel_first=True)
    au_couple_dict = get_zip_ROI_AU()
    au_couple_child = get_AU_couple_child(au_couple_dict)  # AU couple tuple => child fetch list
    au_couple_box = dict()  # value is box (4 tuple coordinate) list

    for AU, AU_couple in au_couple_dict.items():
        au_couple_box[AU_couple] = AU_box_dict[AU]
    box_lst = []
    roi_no_AU_couple_dict = dict()
    roi_no = 0
    for AU_couple, couple_box_lst in au_couple_box.items():
        box_lst.extend(couple_box_lst)
        for _ in couple_box_lst:
            roi_no_AU_couple_dict[roi_no] = AU_couple
            roi_no += 1

    box_lst = np.asarray(box_lst)
    cropped_face = cropped_face.astype(np.float32)
    orig_face = cropped_face
    cropped_face = faster_rcnn.prepare(cropped_face)  # substract mean pixel value
    box_lst = box_lst.astype(np.float32)
    orig_box_lst = box_lst
    batch = [(cropped_face, box_lst), ]
    cropped_face, box_lst = concat_examples(batch, args.device)  # N,3, H, W, ;  N, F, 4

    if box_lst.shape[1] != config.BOX_NUM[args.database]:
        print("error box num {0} != {1}".format(box_lst.shape[1], config.BOX_NUM[args.database]))
        return
    with chainer.no_backprop_mode(), chainer.using_config("train",False):
        cropped_face = chainer.Variable(cropped_face)
        box_lst = chainer.Variable(box_lst)
        roi_preds, _ = faster_rcnn.predict(cropped_face, box_lst)  # R, 22
        roi_feature_maps = faster_rcnn.extract(orig_face, orig_box_lst, 'res5')  # R, 2048 7,7

        roi_images = []
        box_lst = box_lst[0].data.astype(np.int32)
        for box in box_lst:
            y_min, x_min, y_max, x_max = box
            roi_image = orig_face[:, y_min:y_max+1, x_min:x_max+1]  # N, 3, roi_H, roi_W
            roi_images.append(roi_image) # list of  N, 3, roi_H, roi_W
        cmap = plt.get_cmap('jet')
        # image_activate_map = np.zeros((cropped_face.shape[2], cropped_face.shape[3]), dtype=np.float32)
        for box_id, (roi_image, roi_feature_map) in enumerate(zip(roi_images, roi_feature_maps)):
            y_min, x_min, y_max, x_max = box_lst[box_id]
            # 22, roi_h, roi_w, 3
            xp = chainer.cuda.get_array_module(roi_feature_map)
            roi_feature_map = xp.expand_dims(roi_feature_map, 0)
            #   class_roi_overlay_img = 22, roi_h, roi_w
            class_roi_activate_img = heatmap_gen.generate_activate_roi_map(roi_feature_map, (y_max-y_min+1, x_max-x_min+1))
            roi_pred = roi_preds[box_id]  # 22
            # choice_activate_map = np.zeros((y_max-y_min+1, x_max-x_min+1), dtype=np.float32)
            # use_choice = False
            if len(np.nonzero(roi_pred)[0]) > 0: # TODO : 还要做做 class的选择，以及 heatmap采用cv2.add的模式相加
                class_idx = random.choice(np.nonzero(roi_pred)[0])
                AU = config.AU_SQUEEZE[class_idx]
                print(AU)
                choice_activate_map = class_roi_activate_img[class_idx]  # roi_h, roi_w
                activation_color_map = np.round(cmap(choice_activate_map)[:, :, :3]*255).astype(np.uint8)
                overlay_img = roi_images[box_id] / 2 + activation_color_map.transpose(2,0,1) / 2
                overlay_img = np.transpose(overlay_img, (1, 2, 0)).astype(np.uint8)
                vis_img = cv2.cvtColor(overlay_img,cv2.COLOR_RGB2BGR)
                cv2.imshow("new", vis_img)
                cv2.waitKey(0)
            #     use_choice = True
            # if use_choice:
            #     for roi_y_idx, y_idx in enumerate(range(y_min, y_max+1)):
            #         for roi_x_idx, x_idx in enumerate(range(x_min, x_max+1)):
            #             old_val = image_activate_map[y_idx, x_idx]
            #             if old_val < choice_activate_map[roi_y_idx, roi_x_idx]:
            #                 image_activate_map[y_idx, x_idx] = choice_activate_map[roi_y_idx, roi_x_idx]



        # activation_color_map = np.round(cmap(image_activate_map)[:, :, :3] * 255).astype(np.uint8)
        # activation_color_map = activation_color_map.transpose(2,0,1)
        # overlay_img = orig_face / 2 + activation_color_map / 2
        #
        # overlay_img = np.transpose(overlay_img, (1,2,0)).astype(np.uint8)
        # cv2.imshow("new", overlay_img)
        # cv2.waitKey(0)
    # roi_preds = roi_preds.reshape(box_lst.shape[1], len(config.AU_SQUEEZE))  # shape = R, Y
    # roi_preds = chainer.cuda.to_cpu(roi_preds)
    # nonzero_idx = np.nonzero(roi_preds)
    # activate_couple = set()
    # for roi_no in nonzero_idx[0]:
    #     activate_couple.add(roi_no_AU_couple_dict[roi_no])
    # roi_activate = list(activate_couple)


    # AUCouple_face_dict = heatmap_gen.generate_AUCouple_ROI_mask_image(database_name=args.database,
    #                                                       img_path=args.image, roi_activate=roi_activate)  # AU couple and face
    # preds = np.bitwise_or.reduce(roi_preds, axis=0)  # shape = Y
    # assert preds.shape[0] == len(config.AU_SQUEEZE)
    # image_pred_label = list()
    # nonzero_preds = np.nonzero(preds)
    # for AU_idx in nonzero_preds[0]:
    #     image_pred_label.append(config.AU_SQUEEZE[AU_idx])
    #
    # show_label_subimage = defaultdict(list)
    # for AU_couple in AUCouple_face_dict.keys():
    #     for AU in AU_couple:
    #         if AU in image_pred_label:
    #             show_label_subimage[AU_couple].append(AU)
    #
    # total_num = len(show_label_subimage)
    # f,ax = plt.subplots(int(math.ceil(total_num/2)),2)
    # del_AU_couple = list()  # remove the effect of "label fetch", remove unrelated AU
    # for AU_couple in AUCouple_face_dict.keys():
    #     if AU_couple not in show_label_subimage:
    #         del_AU_couple.append(AU_couple)
    # for AU_couple in del_AU_couple:
    #     del AUCouple_face_dict[AU_couple]
    # AUCouple_face_dict = list(AUCouple_face_dict.items())
    # assert len(AUCouple_face_dict) == total_num
    # for i in range(total_num):
    #     ax[i//2,i%2].set_title("AU:{}".format(show_label_subimage[AUCouple_face_dict[i][0]]))
    #     img = AUCouple_face_dict[i][1]  # OpenCV is in BGR mode
    #     img = img[:,:,::-1]  # convert BGR to RGB in Channel axis
    #     ax[i//2,i%2].imshow(img)
    # plt.show()

if __name__ == "__main__":
    main()
