import chainer
from collections import defaultdict
from collections_toolkit.ordered_default_dict import DefaultOrderedDict
import numpy as np
from chainer.dataset import iterator

class BatchKeepOrderIterator(iterator.Iterator):

    def __init__(self, dataset, batch_size=-1, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._shuffle = shuffle
        self._order = None
        # True if the epoch is incremented at the last iteration.
        self._repeat = repeat
        self.length = len(dataset)
        # split to global order and local order, local order must keep original order

        self.seq_dict = DefaultOrderedDict(list)
        for idx, sequence_id in enumerate(self.dataset.sequence_name_list):
            self.seq_dict[sequence_id].append(idx)
        self.offsets = []  # offsets is list of list, inner list is each batch index
        if self.batch_size > 0:
            for sequence_id, fetch_idx_lst in self.seq_dict.items():
                self.offsets.extend(
                    [fetch_idx_lst[i:i + self.batch_size] for i in range(0, len(fetch_idx_lst), self.batch_size)])
        else:
            for sequence_id, fetch_idx_lst in self.seq_dict.items():
                self.offsets.append(fetch_idx_lst)
        self.reset()

    @property
    def epoch_detail(self):
        if self.batch_size > 0:
            return self.epoch + self.current_position * self.batch_size / len(self.dataset)
        if self._order is None:
            return self.epoch + sum(len(self.offsets[i]) for i in range(self.current_position)) / len(self.dataset)
        else:
            return self.epoch + sum(len(self.offsets[self._order[i]]) for i in range(self.current_position)) / len(self.dataset)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration
        self._previous_epoch_detail = self.epoch_detail
        i = self.current_position
        if self._order is None:
            batch = self.dataset[self.offsets[i]]
        else:
            batch = self.dataset[self.offsets[self._order[i]]]

        if i + 1 >= len(self.offsets):
            self.current_position = 0
            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i + 1
        return batch

    next = __next__

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        self.offsets = serializer("offsets", self.offsets)
        if self._order is not None:
            try:
                serializer('order', self._order)
            except KeyError:
                serializer('_order', self._order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            if self.batch_size > 0:
                self._previous_epoch_detail = self.epoch + \
                    (self.current_position * self.batch_size - self.batch_size) / len(self.dataset)
            if self._order is None:
                self._previous_epoch_detail = self.epoch + \
                sum(len(self.offsets[i]) for i in range(self.current_position-1)) / len(self.dataset)
            else:
                self._previous_epoch_detail = self.epoch + \
                sum(len(self.offsets[self._order[i]]) for i in range(self.current_position-1)) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

    def reset(self):

        if self._shuffle:
            self._order = np.random.permutation(len(self.offsets))

        else:
            self._order = None

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.
