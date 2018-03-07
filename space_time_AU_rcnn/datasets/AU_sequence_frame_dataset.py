from space_time_AU_rcnn.datasets.AU_dataset import AUDataset
import numpy as np

class AUTimeSequenceDataset(AUDataset):


    def __init__(self, database, fold, split_name, split_index, mc_manager, train_all_data,
                  previous_frame=50, sample_frame=25, train_mode=True, paper_report_label_idx=None):
        super(AUTimeSequenceDataset, self).__init__(database, fold, split_name, split_index, mc_manager, train_all_data)
        self.previous_frame = previous_frame
        self.sample_frame = sample_frame
        self.train_mode = train_mode
        self.paper_report_label_idx = paper_report_label_idx

    def extract_sequence_key(self, img_path):
        return "_".join((img_path.split("/")[-3], img_path.split("/")[-2]))

    def get_example(self, i):
        if i > len(self):
            raise IndexError("Index too large , i = {}".format(i))
        img_path, from_img_path, AU_set, database_name = self.result_data[i]
        sequence_key = self.extract_sequence_key(img_path)
        sequence_images = []
        sequence_boxes = []
        sequence_labels = []
        for fetch_i in range(i-self.previous_frame, i+1):
            if fetch_i < 0:
                continue
            img_path, *_ = self.result_data[fetch_i]
            if self.extract_sequence_key(img_path) != sequence_key:
                continue
            cropped_face, bbox, label = super(AUTimeSequenceDataset, self).get_example(fetch_i)
            sequence_images.append(cropped_face)
            sequence_boxes.append(bbox)
            sequence_labels.append(label)

        sequence_images = np.stack(sequence_images)  # T, C, H, W
        sequence_boxes = np.stack(sequence_boxes)    # T, R, 4
        sequence_labels = np.stack(sequence_labels)  # T, R, 22/12
        if self.paper_report_label_idx:
            sequence_labels = sequence_labels[:, :, self.paper_report_label_idx]

        if self.train_mode and sequence_boxes.shape[0] > self.sample_frame:
            choice_frame = np.random.choice(np.arange(sequence_boxes.shape[0]), size=self.sample_frame, replace=False)
            choice_frame = np.sort(choice_frame)
            sequence_images = sequence_images[choice_frame, ...]
            sequence_boxes = sequence_boxes[choice_frame, ...]
            sequence_labels = sequence_labels[choice_frame, ...]

        return sequence_images, sequence_boxes, sequence_labels