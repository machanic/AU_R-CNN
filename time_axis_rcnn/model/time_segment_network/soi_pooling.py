# Modified work:
# -----------------------------------------------------------------------------
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
# -----------------------------------------------------------------------------

# Original work of _roi_pooling_slice, forward_cpu and backward_cpu:
# -----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------

# Original work of forward_gpu and backward_gpu:
# -----------------------------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see fast-rcnn/LICENSE for details]
# Written by Ross Girshick
# -----------------------------------------------------------------------------

import numpy
import six

from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check


def _roi_pooling_slice(size, stride, max_size, roi_offset):
    start = int(numpy.floor(size * stride))
    end = int(numpy.ceil((size + 1) * stride))

    start = min(max(start + roi_offset, 0), max_size)
    end = min(max(end + roi_offset, 0), max_size)

    return slice(start, end), end - start


class SOIPooling1D(function.Function):

    """RoI pooling over a set of 2d planes."""

    def __init__(self, outw, spatial_scale):
        self.outw = outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, roi_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 3,
            roi_type.dtype == numpy.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 3,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1,))  # retain bottom_rois for backward
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        channels, width = bottom_data.shape[1:]  # B, C, W
        n_rois = bottom_rois.shape[0]
        # `numpy.zeros` needs to be used because the arrays can be
        # returned without having some of its values updated.
        top_data = numpy.zeros((n_rois, channels, self.outw),
                               dtype=numpy.float32)
        self.argmax_data = numpy.zeros(top_data.shape, numpy.int32)

        for i_roi in six.moves.range(n_rois):
            idx, xmin, xmax = bottom_rois[i_roi]
            xmin = int(round(xmin * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))

            roi_width = max(xmax - xmin + 1, 1)
            stridew = 1. * roi_width / self.outw


            for outw in six.moves.range(self.outw):
                slicew, lenw = _roi_pooling_slice(
                    outw, stridew, width, xmin)
                if slicew.stop <= slicew.start:
                    continue
                roi_data = bottom_data[int(idx), :, slicew]\
                    .reshape(channels, -1)
                top_data[i_roi, :, outw] =\
                    numpy.max(roi_data, axis=1)

                # get the max idx respect to feature_maps coordinates
                max_idx_slice = numpy.unravel_index(
                    numpy.argmax(roi_data, axis=1), (lenw))
                max_idx_slice = max_idx_slice[0] + slicew.start
                self.argmax_data[i_roi, :, outw] = max_idx_slice
        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        channels, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels,
                                    self.outw), dtype=numpy.float32)
        self.argmax_data = cuda.cupy.empty(top_data.shape, numpy.int32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_data, float32 spatial_scale, int32 channels,
            int32 width, int32 pooled_width,
            raw float32 bottom_rois
            ''',
            'float32 top_data, int32 argmax_data',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int c = (i / pooled_width) % channels;
            int num = i / pooled_width / channels;

            int roi_batch_ind = bottom_rois[num * 3 + 0];
            int roi_start_w = round(bottom_rois[num * 3 + 1] * spatial_scale);
            int roi_end_w = round(bottom_rois[num * 3 + 2] * spatial_scale);

            // Force malformed ROIs to be 1x1
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);

            float bin_size_w = static_cast<float>(roi_width)
                           / static_cast<float>(pooled_width);


            int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                          * bin_size_w));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                        * bin_size_w));

            // Add roi offsets and clip to input boundaries
            wstart = min(max(wstart + roi_start_w, 0), width);
            wend = min(max(wend + roi_start_w, 0), width);
            bool is_empty = (wend <= wstart);

            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            int maxidx = -1;
            int data_offset = (roi_batch_ind * channels + c) * width;
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = w;
                if (bottom_data[data_offset + bottom_index] > maxval) {
                    maxval = bottom_data[data_offset + bottom_index];
                    maxidx = bottom_index;
                }
            }
            top_data = maxval;
            argmax_data = maxidx;
            ''', 'roi_pooling_2d_fwd'
        )(bottom_data, self.spatial_scale, channels, width,
          self.outw, bottom_rois, top_data,
          self.argmax_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, width = self._bottom_data_shape[1:]
        n_rois = bottom_rois.shape[0]
        bottom_delta = numpy.zeros(self._bottom_data_shape, numpy.float32)

        for i_roi in six.moves.range(n_rois):
            idx, xmin, xmax = bottom_rois[i_roi]
            idx = int(idx)
            xmin = int(round(xmin * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))
            roi_width = max(xmax - xmin + 1, 1)

            stridew = float(roi_width) / float(self.outw)

            # iterate all the w, h (from feature map) that fall into this ROIs
            for w in six.moves.range(xmin, xmax + 1):
                pwstart = int(numpy.floor(float(w - xmin) / stridew))
                pwend = int(numpy.ceil(float(w - xmin + 1) / stridew))

                pwstart = min(max(pwstart, 0), self.outw)
                pwend = min(max(pwend, 0), self.outw)

                for pw in six.moves.range(pwstart, pwend):
                    max_idx_tmp = self.argmax_data[i_roi, :, pw]
                    for c in six.moves.range(channels):
                        if max_idx_tmp[c] == w:
                            bottom_delta[idx, c, w] += \
                                gy[0][i_roi, c, pw]
        return bottom_delta, None

    def backward_gpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff, raw int32 argmax_data, int32 num_rois,
            float32 spatial_scale, int32 channels, int32 width,
            int32 pooled_width, raw float32 bottom_rois
            ''',
            'float32 bottom_diff',
            '''
            int w = i % width;
            int c = (i / width) % channels;
            int num = i / (width  * channels);

            float gradient = 0;
            // Accumulate gradient over all ROIs that pooled this element
            for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
                // Skip if ROI's batch index doesn't match num
                if (num != static_cast<int>(bottom_rois[roi_n * 3])) {
                    continue;
                }
                int roi_start_w = round(bottom_rois[roi_n * 3 + 1]
                                        * spatial_scale);
                int roi_end_w = round(bottom_rois[roi_n * 3 + 2]
                                      * spatial_scale);


                // Skip if ROI doesn't include (h, w)
                const bool in_roi = (w >= roi_start_w && w <= roi_end_w);
                if (!in_roi) {
                    continue;
                }

                int offset = (roi_n * channels + c) * pooled_width;

                // Compute feasible set of pooled units that could have pooled
                // this bottom unit

                // Force malformed ROIs to be 1x1
                int roi_width = max(roi_end_w - roi_start_w + 1, 1);

                float bin_size_w = static_cast<float>(roi_width)
                               / static_cast<float>(pooled_width);

                int pwstart = floor(static_cast<float>(w - roi_start_w)
                                    / bin_size_w);
                int pwend = ceil(static_cast<float>(w - roi_start_w + 1)
                                 / bin_size_w);

                pwstart = min(max(pwstart, 0), pooled_width);
                pwend = min(max(pwend, 0), pooled_width);

                for (int pw = pwstart; pw < pwend; ++pw) {
                    int index_ = pw + offset;
                    if (argmax_data[index_] == w) {
                        gradient += top_diff[index_];
                    }
                }
                
            }
            bottom_diff = gradient;
            ''', 'roi_pooling_2d_bwd'
        )(gy[0], self.argmax_data, bottom_rois.shape[0], self.spatial_scale,
          channels, width, self.outw,
          bottom_rois, bottom_diff)

        return bottom_diff, None


def soi_pooling_1d(x, sois, outw, spatial_scale):
    """Spatial Region of Interest (ROI) pooling function.

    This function acts similarly to :class:`~functions.MaxPooling2D`, but
    it computes the maximum of input spatial patch for each channel
    with the region of interest.

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            3 dimentional: (n: batch, c: channel, w: width).
        sois (~chainer.Variable): Input roi variable. The shape is expected to
            be (n: data size, 3), and each datum is set as below:
            (batch_index, x_min, x_max).
        outw (int): Width of output image after pooled.
        spatial_scale (float): Scale of the roi is resized.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    """
    return SOIPooling1D(outw, spatial_scale)(x, sois)


if __name__ == "__main__":
    import  numpy as np
    import chainer
    import chainer.functions as F
    x = np.arange(16, dtype=np.float32)
    print(x[15])
    print(x)
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, 0)

    sois = np.array([[0, 0, 7], [0, 2, 10]], dtype=np.float32)  # shape = 1, 2, 3
    outw = 3
    spatial_scale = 1.0

    x = chainer.cuda.to_gpu(x, 0)
    sois = chainer.cuda.to_gpu(sois, 0)
    o = soi_pooling_1d(chainer.Variable(x), chainer.Variable(sois),outw=outw, spatial_scale=1.0)
    print(o)

    import cupy as cp
    x = cp.expand_dims(x, 2)  # 1, 1, 1, 16
    sois = cp.array([[0, 0, 0, 7, 0], [0, 2,0, 10, 0]], dtype=np.float32)
    outw= 3
    outh = 1
    o = F.roi_pooling_2d(x, sois, outh, outw, spatial_scale=1.0)


    print(o)