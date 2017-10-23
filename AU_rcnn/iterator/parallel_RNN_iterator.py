from __future__ import division
import numpy as np
from collections import defaultdict
import config
from multiprocessing import Pool, Manager
import multiprocessing
import chainer.iterators.multiprocess_iterator
from overrides import overrides
import heapq

def _prefetch(cls_instance, queue_prefetch, lock):
    print("_prefetch")
    return cls_instance._prefetch(queue_prefetch, lock)


def _sort_queue(cls_instance, queue_prefetch, queue_fetch, lock):
    print("sort_queue")
    return cls_instance._sort_queue(queue_prefetch, queue_fetch, lock)

class ParallelRNNIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True, n_processes=10, n_prefetch=10):
        self._last_signal = object()
        self.dataset = dataset
        self.n_prefetch = n_prefetch
        self.n_processes = n_processes
        self._workers = []
        self.shared_attrib = set(["current_bucket","iteration","bucket_video_select_idx","proceed_video","_previous_epoch_detail","epoch","is_new_epoch","epoch_detail"])
        self.is_new_epoch = False
        self.repeat = repeat

        self.video_offset = self.dataset.video_offset
        self.video_bucket = self.dataset.video_bucket
        self.video_count = self.dataset.video_count

        self.start_offsets_bucket = defaultdict(list)  # each bucket has different batch_size, value is list of video offset start index
        self.bucket_max_seq_count = dict()
        self.video_padding_offset = dict()
        self.batch_size = batch_size

        for bucket_label, video_id_list in self.video_bucket.items():
            # 该桶子最长的视频多少帧
            self.bucket_max_seq_count[bucket_label] = max([(video_id, self.video_count[video_id]) \
                                                          for video_id in video_id_list], key=lambda e:e[1])
            # assert batch_size <= len(video_id_list) # batch_size < video_id_list,桶子内切换文章，一个桶子处理完了切换桶子
            for video_id in video_id_list:
                self.video_padding_offset[video_id] = self.video_offset[video_id] + self.video_count[video_id]
                start_offset = self.video_offset[video_id]
                self.start_offsets_bucket[bucket_label].append((video_id, start_offset)) # value is list of video offset start index
        self.iteration = {bucket_label: 0 for bucket_label in self.video_bucket.keys()}  # 指各桶子内部的齐头并进的指针情况
        self.bucket_video_select_idx = {bucket_label: 0 for bucket_label in
                                        self.video_bucket.keys()}  # 处理到该桶子内哪篇文章了，用于判断是否要切换桶子
        self.current_bucket = 0  # forever ++
        self.proceed_video = 0   # forever ++
        self.epoch = 0
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.

        self.all_videos_count = sum(len(video_id_list) for video_id_list in self.video_bucket.values())
        self.n_fetch = self.n_prefetch//2

    @overrides
    def finalize(self):
        self.queue_fetch.put((self._last_signal,-1))
        self.queue_prefetch.put((self._last_signal,-1))

        for worker in self._workers:
            worker.join()

        self.queue_fetch.close()
        self.queue_fetch.join_thread()
        self.queue_prefetch.close()
        self.queue_prefetch.join_thread()


    def run_parallel(self):
        manager = Manager()
        self.queue_prefetch = multiprocessing.Queue(maxsize=self.n_prefetch)
        self.queue_fetch = multiprocessing.Queue(maxsize=self.n_prefetch)
        self.lock1 = multiprocessing.Lock()
        self.lock2 = multiprocessing.Lock()
        self.shared_dict = self.create_shared_dict(manager)
        # pool = Pool()
        for _ in range(self.n_processes):
            worker = multiprocessing.Process(target=self._prefetch,
                                             args=(self.shared_dict, self.queue_prefetch, self.lock1))
            worker.daemon = True
            self._workers.append(worker)
            worker.start()
            # handler = pool.apply_async(_prefetch, args=(self, self.queue_prefetch, self.lock1))
            # print("appy_async")
        worker = multiprocessing.Process(target=self._sort_queue,
                                         args=(self.shared_dict, self.queue_prefetch, self.queue_fetch, self.lock1))
        worker.daemon = True
        self._workers.append(worker)
        worker.start()


        # handler = pool.apply_async(_sort_queue, args=(self, self.queue_prefetch, self.queue_fetch, self.lock2))
        # pool.close()
        # self.pool.join()
        # self.queue_prefetch.close()
        # self.queue_prefetch.join_thread()

    def _sort_queue(self, shared_dict, queue_prefetch, queue_fetch, lock1):

        sorted_list = list()
        while True:
            print("before sort_queue lock")
            lock1.acquire()
            print("after sort_queue lock")
            self.merge_back_from_shared_dict(shared_dict)
            lock1.release()

            epoch = self.current_bucket // len(self.video_bucket)
            if not self.repeat and epoch >= len(self.video_bucket):
                break

            mini_batch, mini_batch_sort_id = queue_prefetch.get(block=True)
            print("_sort_queue get!!  queue_prefetch now size: {0} , max_qsize:{1}".format(queue_prefetch.qsize(), self.n_prefetch))
            if mini_batch is self._last_signal:
                break
            if len(sorted_list) < self.n_fetch:
                sorted_list.append((mini_batch_sort_id, mini_batch))
            else:
                sorted_list.sort(key=lambda e:e[0])
                for entry in sorted_list:
                    _mini_batch_id, _mini_batch = entry
                    print("sorted : {}".format(_mini_batch_id))
                    queue_fetch.put((_mini_batch, _mini_batch_id), block=False)
                sorted_list.clear()
        queue_prefetch.close()
        queue_prefetch.join_thread()
        queue_fetch.close()
        queue_fetch.join_thread()


    def create_shared_dict(self, manager):
        if manager is None:
            manager = Manager()
        shared_dict = manager.dict()
        for k,v in self.__dict__.items():
            if k in self.shared_attrib:
                shared_dict[k] = v

        return shared_dict

    def merge_back_from_shared_dict(self, shared_dict):
        for k,v in shared_dict.items():
            if k in self.__dict__:
                setattr(self, k, v)

    def disperse_shared_dict(self, shared_dict):
        for k, v in self.__dict__.items():
            if k in self.shared_attrib:
                shared_dict[k] = v


    def _prefetch(self, shared_dict, queue_prefetch, lock):

        while True:
            print("before prefetch lock1")
            lock.acquire()
            print("after prefetch lock1")
            self.merge_back_from_shared_dict(shared_dict)

            current_bucket = self.current_bucket % len(self.video_bucket)
            max_video_id, max_seq_count = self.bucket_max_seq_count[current_bucket]
            if self.iteration[current_bucket] >= max_seq_count:  # FIXME if max_seq_count == 1, this will cause bug
                self.iteration[current_bucket] = 0

                bucket_video_select_start_idx = self.bucket_video_select_idx[current_bucket]
                offsets = self.start_offsets_bucket[current_bucket][bucket_video_select_start_idx: \
                    bucket_video_select_start_idx + self.batch_size]
                self.proceed_video += len(offsets)
                if self.bucket_video_select_idx[current_bucket] + self.batch_size >= len(self.start_offsets_bucket[current_bucket]):
                    self.current_bucket += 1  # forever ++
                self.bucket_video_select_idx[current_bucket] += self.batch_size

            self.iteration[current_bucket] += 1
            self.disperse_shared_dict(shared_dict)
            lock.release()

            mini_batch, mini_batch_sort_id = self.get_batch()  # it is tuple: mini_batch, mini_batch_sort_id
            print("before prefetch lock2")
            lock.acquire()
            print("after prefetch lock2")
            self.merge_back_from_shared_dict(shared_dict)
            self._previous_epoch_detail = self.epoch_detail

            epoch = self.current_bucket // len(self.video_bucket)
            if not self.repeat and epoch >= len(self.video_bucket):
                lock.release()
                break
            self.is_new_epoch = self.epoch < epoch
            if self.is_new_epoch:
                self.epoch = epoch
                for bucket in self.bucket_video_select_idx.keys():
                    self.bucket_video_select_idx[bucket] = 0
                    self.iteration[bucket] = 0
            lock.release()  #FIXME

            queue_prefetch.put((mini_batch, mini_batch_sort_id), block=True)
            print("before prefetch lock3")
            lock.acquire()
            print("after prefetch lock3")
            self.disperse_shared_dict(shared_dict)
            print("queue_prefetch now size: {0} , max_qsize:{1}".format(queue_prefetch.qsize(),self.n_prefetch))
            print("queue_sort now size: {0} , max_qsize:{1}".format(self.queue_fetch.qsize(), self.n_prefetch))
            lock.release()
        queue_prefetch.close()
        queue_prefetch.join_thread()


    def __next__(self):
        next_batch = self.queue_fetch.get(block=True)
        print("getting next batch")
        return next_batch

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    @property
    def epoch_detail(self):
        return self.proceed_video / self.all_videos_count

    def get_batch(self):
        mini_batch = []
        current_bucket = self.current_bucket % len(self.video_bucket)
        max_video_id, max_seq_count = self.bucket_max_seq_count[current_bucket]
        bucket_video_select_start_idx = self.bucket_video_select_idx[current_bucket]
        offsets = self.start_offsets_bucket[current_bucket][bucket_video_select_start_idx: \
                                                            bucket_video_select_start_idx + self.batch_size]

        for video_id, start_offset in offsets:  # len(offsets) = batch_size
            current_offset = start_offset + self.iteration[current_bucket] - 1
            if current_offset >= self.video_padding_offset[video_id]:
                assert video_id != max_video_id
                mini_batch.append((np.zeros(config.CHANNEL, config.IMG_SIZE[0], config.IMG_SIZE[1]),
                                           np.array([[0.0,0.0, config.IMG_SIZE[0], config.IMG_SIZE[1]]],dtype=np.float32),
                                           np.zeros(shape=(1, len(config.AU_SQUEEZE)), dtype=np.int32)))
            else:
                try:
                    mini_batch.append(self.dataset[current_offset])
                except IndexError:
                    mini_batch.append(self.dataset[current_offset - 1])  # may cause bug?
                    # all_frame_in_batch.append((np.zeros(config.CHANNEL, config.IMG_SIZE[0], config.IMG_SIZE[1]),
                    #                            np.array([[0.0, 0.0, config.IMG_SIZE[0], config.IMG_SIZE[1]]],
                    #                                     dtype=np.float32),
                    #                            np.zeros(shape=(1, len(config.AU_SQUEEZE)), dtype=np.int32)))
        mini_batch_sort_id = "{0}/{1}/{2}".format(self.current_bucket, bucket_video_select_start_idx,
                                                  self.iteration[current_bucket])
        return mini_batch, mini_batch_sort_id

    def serialize(self, serializer):
        self.merge_back_from_shared_dict(self.shared_dict)
        self.current_bucket = serializer('current_bucket', self.current_bucket)
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        self.proceed_video = serializer("proceed_video", self.proceed_video)
        self.bucket_video_select_idx = serializer("bucket_video_select_idx", self.bucket_video_select_idx)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                                          (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.




