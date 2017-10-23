import chainer
import six
import numpy as np
import os

class RemoveNonLabelMultiprocessIterator(chainer.iterators.MultiprocessIterator):

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True,
                 n_processes=None, n_prefetch=1, shared_mem=None, is_sigmoid_cross_entropy=True, remove_non_label=True,
                 label_balance_dict_path=None, need_balance=True):
        super(RemoveNonLabelMultiprocessIterator, self).__init__(dataset, batch_size, repeat, shuffle,
                                                                 n_processes, n_prefetch, shared_mem)
        self.is_sigmoid_cross_entropy = is_sigmoid_cross_entropy
        self.remove_non_label = remove_non_label
        self.label_balance_dict = dict()
        if os.path.exists(label_balance_dict_path):
            print("label balance dict:{} load".format(label_balance_dict_path))
            with open(label_balance_dict_path, "r") as file_obj:  # note that you have to pass label_balance_dict_path according to sigmoid or cross entropy
                for line in file_obj:
                    line = line.strip().split()
                    self.label_balance_dict[line[0]] = int(line[1])
        self.need_balance = need_balance

    def _get(self):
        n = len(self.dataset)
        i = self.current_position

        batch = []

        for _ in six.moves.range(self.batch_size):
            d = self._ordered_data_queue.get()
            if d is self._last_signal:
                break
            cropped_face, bboxes, labels = d
            if self.is_sigmoid_cross_entropy and self.remove_non_label:
                all_label_zero = not np.any(labels)
                if all_label_zero:
                    continue
            elif not self.is_sigmoid_cross_entropy and self.remove_non_label:
                if labels == 0:
                    continue
            if self.need_balance:
                labels = labels.tolist()
                bboxes = bboxes.tolist()
                for idx, label in enumerate(labels):
                    label_key = ",".join(map(str, label))
                    replicate = self.label_balance_dict.get(label_key, 1)  #  注意label_balance_dict应该根据模式传入正确的
                    for _ in range(replicate-1):
                        labels.append(label)
                        bboxes.append(bboxes[i])
                bboxes = np.stack(bboxes).astype(np.float32)
                label = np.stack(label).astype(np.int32)
            batch.append((cropped_face, bboxes, labels))
            i += 1
            if i >= n:
                self.epoch += 1
                self.is_new_epoch = True
                i = 0
                if not self._repeat:
                    break
        self.current_position = i
        # Eventually overwrite the (possibly shuffled) order.
        self._order = self._prefetch_order
        return batch