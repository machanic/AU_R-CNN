import copy
import time

import chainer.training.extensions
import numpy as np
from chainer import Reporter

import config


class SpeedEvaluator(chainer.training.extensions.Evaluator):
    trigger = 1, "epoch"
    default_name = "AU_RCNN_validation"
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self,  iterator, target, concat_example_func, device, trail_times, each_trail_iteration, database):
        super(SpeedEvaluator, self).__init__(iterator, target, converter=concat_example_func,device=device)
        self.trail_times = trail_times
        self.each_trail_iteration = each_trail_iteration
        self.database = database

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        tot_trail_time = []
        for trail_time in range(self.trail_times):
            print("trail in {}".format(trail_time))
            it = copy.copy(iterator)
            each_trail_time = []
            for idx, batch in enumerate(it):
                if idx >= self.each_trail_iteration:
                    break
                batch = self.converter(batch, device=self.device)

                imgs, _, _ = batch
                imgs = chainer.Variable(imgs)
                before_time = time.time()
                preds = target.predict(imgs)  # R', class_num
                each_trail_time.append(time.time() - before_time)
            tot_trail_time.append(np.mean(np.array(each_trail_time)))
        mean_time_elapse = np.mean(tot_trail_time)
        standard_var_time_elapse = np.var(tot_trail_time)
        reporter = Reporter()
        reporter.add_observer("main", target)
        observation = {"mean": mean_time_elapse, "var" : standard_var_time_elapse}
        print(observation)
        with reporter.scope(observation):
            reporter.report(observation, target)
        return observation