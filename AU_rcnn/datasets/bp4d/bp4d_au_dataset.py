from AU_rcnn.utils import read_image
import chainer
import os
from collections import defaultdict, OrderedDict
import numpy as np
from dataset_toolkit.compress_utils import get_zip_ROI_AU
import cv2






class BP4DDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dir, split_name="train", split_index=1):
        self.dir = dir
        self.split_name = split_name
        au_couple_dict = get_zip_ROI_AU()
        self.img_dir = self.dir + "/BP4D_crop/"
        self.AU_occur_dir = self.dir + "/AUCoding/"
        self.AU_intensity_dir = self.dir + "/AU-Intensity-Codes3.0/"
        self.AU_mask_dir = self.dir + "/BP4D_AUmask/"
        self._video_seq = defaultdict(dict)  # subject+"/"+emotion_seq => {frame_no : absolute_path_list}
        self.AU_occur_label = {}  # subject + "/" + emotion_seq + "/" + frame => {"1":1, "2":0} which is "AU": 1/0
        self.AU_intensity_label = {}  # subject + "/" + emotion_seq + "/" + frame => ... not implemented
        id_list_file = os.path.join(self.dir + "/idx/", "id_{0}_{1}.txt".format(split_name, split_index))

        # id list file path
        # 需要先构造一个list的txt文件:id_trainval_0.txt, 每一行是subject + "/" + emotion_seq + "/" frame

        self.ids = [id_.strip() for id_ in open(id_list_file) if id_.strip()] # [subject + "/" + emotion_seq + "/" frame, ...] frame带leading 0
        _temp_images = []  # 为了按照ids文件重排序
        _temp_labels = []
        _temp_au_mask = []

        ids_order_dict = OrderedDict()
        for index, id_ in enumerate(self.ids):
            ids_order_dict[id_] = index

        self.ids_dir = set()
        for id in self.ids:
            ids_dir = os.path.dirname(id) # remove frame
            self.ids_dir.add(ids_dir) # unique folder store in
        AU_occur_path_lst = ["{0}/{1}.csv".format(self.AU_occur_dir, subject_seq.replace("/", "_")) \
                                            for subject_seq in self.ids_dir]

        for file_name in AU_occur_path_lst:
            subject_name = file_name[file_name.rindex("/")+1:].split("_")[0]
            sequence_name = file_name[file_name.rindex("_")+1:file_name.rindex(".")]
            # obtain image path first, which folder path comes from au_occur filename
            video_dir = self.img_dir + os.sep + subject_name + os.sep + sequence_name + os.sep
            frame_dict = {} # "1" : "001"
            for img_file in sorted(os.listdir(video_dir)):
                frame = img_file[:img_file.rindex(".")]
                self._video_seq[subject_name+"/"+sequence_name][int(frame)] = video_dir+img_file
                frame_dict[int(frame)] = frame
            AU_column_idx = {}
            with open(file_name, "r") as au_file_obj:
                for idx, line in enumerate(au_file_obj):
                    if idx == 0: # header specify Action Unit
                        for col_idx, AU in enumerate(line.split(",")[1:]):
                            AU_column_idx[AU] = col_idx + 1  # read header
                        continue  # read head over , continue

                    lines = line.split(",")
                    frame = int(lines[0])


                    if frame not in frame_dict:
                        print("error frame not found! {0}".format(subject_name+"/"+sequence_name+"/"+lines[0]))
                        continue
                    # only obtain 1/0 label, not 9 unknown to train
                    au_label_dict = {AU: int(lines[AU_column_idx[AU]]) for AU in AU_ROI.keys() \
                                     if int(lines[AU_column_idx[AU]]) != 9}  # "AU":1 or "AU":0
                    # key AU != 0 because AU_ROI don't contain key = 0
                    au_mask_dict = {AU: "{0}/{1}/{2}/{3}_AU_{4}.png".format(self.AU_mask_dir,
                                                                            subject_name, sequence_name,
                                                                            frame_dict[frame], ",".join(au_couple_dict[AU])) \
                                                                            for AU in au_label_dict.keys()} # "AU": mask_path
                    self.AU_occur_label[subject_name+"/"+sequence_name+ "/" + frame_dict[int(frame)]] = au_label_dict
                    if int(frame) not in self._video_seq[subject_name+"/"+sequence_name]:
                        print("image frame not fonund!", subject_name + "/" + sequence_name + "/" + frame)
                        continue
                    _temp_images.append(self._video_seq[subject_name + "/" + sequence_name][int(frame)])
                    _temp_labels.append(au_label_dict)
                    _temp_au_mask.append(au_mask_dict)

        self._images = np.full((len(self.ids),), None, dtype=object)
        self._labels = np.full((len(self.ids),), None, dtype=object)
        self._au_mask = np.full((len(self.ids),), None, dtype=object)
        # reorder by self.ids
        for img_idx, img_path in enumerate(_temp_images):
            subject_name = img_path.split(os.sep)[-3]
            sequence_name = img_path.split(os.sep)[-2]
            img_file = img_path.split(os.sep)[-1]
            frame = img_file[:img_file.rindex(".")]
            index = ids_order_dict["{0}/{1}/{2}".format(subject_name,sequence_name,frame)]


            with open("/tmp/error_mask.log", "a") as err_mask:
                for AU, mask_path in list(_temp_au_mask[img_idx].items()):
                    if not os.path.exists(mask_path):
                        print("mask file {} not exists!".format(_temp_au_mask[img_idx][AU]))
                        err_mask.write("{}\n".format(_temp_au_mask[img_idx][AU]))
                        del _temp_au_mask[img_idx][AU]
                err_mask.flush()


            if len(_temp_au_mask[img_idx]) > 0:
                self._images[index] = img_path
                self._labels[index] = _temp_labels[img_idx]
                self._au_mask[index] = _temp_au_mask[img_idx]
        self._images = self._images[self._images != np.array(None)] # remove None value
        self._labels = self._labels[self._labels != np.array(None)]
        self._au_mask = self._au_mask[self._au_mask != np.array(None)]
        self._num_examples = self._images.size

    def __len__(self):
        return self._num_examples

    def get_example(self, i):
        '''
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        :param i:  the index of the example
        :return: tuple of an image and its all bounding box
        '''
        if i > len(self):
            raise IndexError("Index too large")
        # print("getting idx={0}, img={1}".format(i, self._images[i]))
        img = read_image(self._images[i], color=True)
        label = []
        bbox = []
        err_mask = open("/tmp/error_mask.log", "a")
        for AU, mask_path in self._au_mask[i].items():
            # if self.split_name == 'test':  # 做metric evaluation时候只读取label文件有的AU的mask，但训练时需要训练背景
            #     if self._labels[i][AU] == 0:
            #         continue  # 如果有个test的全都是背景，那么可能造成bbox一个都没有的情况，进而报错
            mask = read_image(mask_path, dtype=np.uint8, color=False)

            # use connectivity components to seperate polygon
            connect_arr = cv2.connectedComponents(mask[0], connectivity=4, ltype=cv2.CV_32S)
            component_num = connect_arr[0]
            label_matrix = connect_arr[1]
            # print("mask: {0} connect_num:{1}".format(mask_path, component_num))
            # print("img: {0}, mask:{1} label:{2}".format(self._images[i], mask_path, self._labels[i][AU]))

            # convert mask polygon to rectangle

            for component_label in range(1, component_num):

                row_col = list(zip(*np.where(label_matrix==component_label)))
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
                coordinates = (y_min,x_min,y_max,x_max)


                if y_min == y_max and x_min == x_max:  # 尖角处会产生孤立的单个点，会不会有一个mask只有尖角？
                    # print(("single point mask: img:{0} mask:{1}".format(self._images[i], mask_path)))
                    err_mask.write(("single point mask: img:{0} mask:{1}\n".format(self._images[i], mask_path)))
                    continue
                if coordinates not in bbox:
                    bbox.append(coordinates)
                    # FIXME should convert multi-label to [1,1,0,0,1] because each region may have multiple labels? but CRF can't deal with that case yet
                    if self._labels[i][AU] == 1:
                        label.append(int(AU))
                    else:
                        label.append(0)
                # FIXME don't support multi-label in same region
                # same region but different AU label, labeled as 0 before
                elif label[bbox.index(coordinates)] == 0 and self._labels[i][AU] != 0: # already have this bbox coordinate but labeled as 0 before
                    label[bbox.index(coordinates)] = int(AU) # same bounding box should use the foreground AU label
            del label_matrix
            del mask

        err_mask.flush()
        err_mask.close()
        if len(bbox) == 0:
            with open("/tmp/error_mask.log", "a") as err_mask:
                print("!!!! bbox=0 len, img_path:{0}, mask_path:{1}".format(self._images[i], self._au_mask[i]))
                err_mask.write("!!!! bbox=0 len, img_path:{0}, mask_path:{1}\n".format(self._images[i], self._au_mask[i]))
                err_mask.flush()
                raise IndexError()
        bbox = np.stack(bbox).astype(np.float32)  # #FIXME 这句话会报错，有可能会出现1个都没有的情况
        label = np.stack(label).astype(np.int32)
        assert bbox.shape[0] == label.shape[0]
        if self.split_name == 'test':
            return (img, bbox), label, bbox  # 没办法之举，因为chainercv.utils.apply_prediction_to_iterator需要第一个元组传入faster_rcnn的predict函数
        return img, bbox, label