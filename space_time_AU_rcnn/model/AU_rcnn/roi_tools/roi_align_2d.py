# -*- coding: utf-8 -*-
# Mask R-CNN
# Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
# https://arxiv.org/abs/1703.06870

import numpy
import six

from chainer import cuda
from chainer import function
from chainer import function_node
from chainer.utils import type_check


class ROIAlign2D(function.Function):
    """RoI align over a set of 2d planes."""

    def __init__(self, outh, outw, spatial_scale):
        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, roi_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            roi_type.dtype == numpy.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 5,
        )

    def forward_cpu(self, inputs):
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        # `numpy.zeros` needs to be used because the arrays can be
        # returned without having some of its values updated.
        top_data = numpy.zeros(
            (n_rois, channels, self.outh, self.outw), dtype=numpy.float32)

        for i_roi in six.moves.range(n_rois):
            idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
            xmin = xmin * self.spatial_scale
            xmax = xmax * self.spatial_scale
            ymin = ymin * self.spatial_scale
            ymax = ymax * self.spatial_scale
            roi_width = max(xmax - xmin, 1.)
            roi_height = max(ymax - ymin, 1.)
            strideh = 1. * roi_height / self.outh
            stridew = 1. * roi_width / self.outw

            for outh in six.moves.range(self.outh):
                for outw in six.moves.range(self.outw):

                    cy = (outh + 0.5) * strideh + ymin
                    cx = (outw + 0.5) * stridew + xmin

                    x00 = numpy.array((cy, cx), dtype=numpy.float32)
                    p, q = x00 - numpy.floor(x00)
                    bound = (height - 1, width - 1)
                    x0 = numpy.maximum(numpy.floor(x00 - 0.0), (0, 0)).astype(
                        numpy.int32)
                    x1 = numpy.minimum(x0 + (1, 1), bound).astype(numpy.int32)

                    #print('({}, {}) を計算するのに、featuremapの({}, {}) を参照し、 ({}, {}), ({}, {}), {}, {}'.format(outw, outh, cx, cy, x0[0], x0[1], x1[0], x1[1], q, p))

                    roi_data = bottom_data[int(idx), :, x0[0], x0[1]] * (1-p)*(1-q) \
                                + bottom_data[int(idx), :, x1[0], x0[1]] * p * (1-q) \
                                + bottom_data[int(idx), :, x0[0], x1[1]] * (1-p) * q \
                                + bottom_data[int(idx), :, x1[0], x1[1]] * p * q
                    top_data[i_roi, :, outh, outw] = roi_data
        return top_data,

    def forward_gpu(self, inputs):
        # ROIの情報だけは取っておく -> backwardのinputsに来るからいらないのか？
        self.retain_inputs((1,))

        self._bottom_data_shape = inputs[0].shape
        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty(
            (n_rois, channels, self.outh, self.outw), dtype=numpy.float32)
        cuda.cupy.ElementwiseKernel('''
            raw float32 bottom_data, float32 spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            raw float32 bottom_rois
            ''', 'float32 top_data', '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int num = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_rois[num * 5 + 0];
            float roi_start_w = bottom_rois[num * 5 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[num * 5 + 2] * spatial_scale;
            float roi_end_w = bottom_rois[num * 5 + 3] * spatial_scale;
            float roi_end_h = bottom_rois[num * 5 + 4] * spatial_scale;

            float roi_width = max(roi_end_w - roi_start_w, 1.f);
            float roi_height = max(roi_end_h - roi_start_h, 1.f);
            float bin_size_h = roi_height
                           / static_cast<float>(pooled_height);
            float bin_size_w = roi_width
                           / static_cast<float>(pooled_width);

            // binのインデックスph,pwからfeature map上の座標cy, cxへ
            // 更にいうと、cy,cxは各ビンの中心の座標を指している
            float cy = (ph + 0.5) * bin_size_h + roi_start_h;
            float cx = (pw + 0.5) * bin_size_w + roi_start_w;
            float p = cy - floor(cy);
            float q = cx - floor(cx);
            int y1 = max(static_cast<int>(floor(cy-0.0)), 0);
            int x1 = max(static_cast<int>(floor(cx-0.0)), 0);
            int y2 = min(y1 + 1, height-1);
            int x2 = min(x1 + 1, width-1);

            // このカーネルが処理するチャンネルへのオフセット
            int data_offset = (roi_batch_ind * channels + c) * height * width;
            float val = bottom_data[data_offset + y1 * width + x1] * (1-p) * (1-q);
            val += bottom_data[data_offset + y2 * width + x1] * p * (1-q);
            val += bottom_data[data_offset + y1 * width + x2] * (1-p) * q;
            val += bottom_data[data_offset + y2 * width + x2] * p * q;
            top_data = val;
            ''', 'roi_align_2d_fwd')(bottom_data, self.spatial_scale, channels,
                                     height, width, self.outh, self.outw,
                                     bottom_rois, top_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        n_rois = bottom_rois.shape[0]
        bottom_delta = numpy.zeros(self._bottom_data_shape, numpy.float32)

        for i_roi in six.moves.range(n_rois):
            idx, xmin, ymin, xmax, ymax = bottom_rois[i_roi]
            idx = int(idx)
            xmin = xmin * self.spatial_scale
            xmax = xmax * self.spatial_scale
            ymin = ymin * self.spatial_scale
            ymax = ymax * self.spatial_scale
            roi_width = max(xmax - xmin, 1)
            roi_height = max(ymax - ymin, 1)

            strideh = float(roi_height) / float(self.outh)
            stridew = float(roi_width) / float(self.outw)

            #gyの各成分に対して、対応する4近傍点にgradを加算する
            _, c, rows, cols = gy[0].shape
            for y in range(rows):
                for x in range(cols):
                    cy = (y + 0.5) * strideh + ymin
                    cx = (x + 0.5) * stridew + xmin

                    x00 = numpy.array((cy, cx), dtype=numpy.float32)
                    p, q = x00 - numpy.floor(x00)
                    bound = (height - 1, width - 1)
                    x0 = numpy.maximum(numpy.floor(x00 - 0.0), (0, 0)).astype(
                        numpy.int32)
                    x1 = numpy.minimum(x0 + (1, 1), bound).astype(numpy.int32)

                    bottom_delta[idx, :, x0[0], x0[1]] += (1 - p) * (
                        1 - q) * gy[0][i_roi, :, y, x]
                    bottom_delta[idx, :, x1[0], x0[1]] += p * (
                        1 - q) * gy[0][i_roi, :, y, x]
                    bottom_delta[idx, :, x0[0], x1[1]] += (
                        1 - p) * q * gy[0][i_roi, :, y, x]
                    bottom_delta[idx, :, x1[0], x1[1]] += p * q * gy[0][
                        i_roi, :, y, x]

        return bottom_delta, None

    def backward_gpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)
        cuda.cupy.ElementwiseKernel('''
            raw float32 top_diff, int32 num_rois,
            float32 spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width, raw float32 bottom_rois
            ''', 'float32 bottom_diff', '''
            int w = i % width;
            int h = (i / width) % height;
            int c = (i / (width * height)) % channels;
            int num = i / (width * height * channels);

            float gradient = 0;
            // feature map上のある1ピクセルに伝播するgradを計算する、のがこのカーネルの目的
            // 各roi、各ビンに対して、もしこのカーネルが担当するfeature map上のピクセルが関与していた場合、
            // gradを加算する
            for (int roi_n = 0; roi_n < num_rois; ++roi_n) {

                // Skip if ROI's batch index doesn't match num
                if (num != static_cast<int>(bottom_rois[roi_n * 5])) {
                    continue;
                }

                // float精度でのfeature map上のROIの開始位置、終了位置
                float roi_start_w = bottom_rois[roi_n * 5 + 1]
                                        * spatial_scale;
                float roi_start_h = bottom_rois[roi_n * 5 + 2]
                                        * spatial_scale;
                float roi_end_w = bottom_rois[roi_n * 5 + 3]
                                      * spatial_scale;
                float roi_end_h = bottom_rois[roi_n * 5 + 4]
                                      * spatial_scale;

                int offset = (roi_n * channels + c) * pooled_height
                             * pooled_width;

                // Compute feasible set of pooled units that could have pooled
                // this bottom unit
                float roi_width = max(roi_end_w - roi_start_w, 1.f);
                float roi_height = max(roi_end_h - roi_start_h, 1.f);

                // Skip if ROI doesn't include (h, w)
                const bool in_roi = (roi_start_w-2<=w && w<=roi_end_w+2 && roi_start_h-2<=h && h<=roi_end_h+2);
                if (!in_roi) {
                    continue;
                }

                float bin_size_h = roi_height
                               / static_cast<float>(pooled_height);
                float bin_size_w = roi_width
                               / static_cast<float>(pooled_width);

                // 各ビンに対して、feature map上のfloat精度のy, xを得る
                for (int row=0; row<pooled_height; ++row) {
                    for (int col=0; col<pooled_width; ++col) {
                        float cx = (col + 0.5f) * bin_size_w + roi_start_w;
                        float cy = (row + 0.5f) * bin_size_h + roi_start_h;
                        // これを計算するために使った4近傍点に対して、gradを加算する
                        float p = cy - floor(cy);
                        float q = cx - floor(cx);
                        int x0 = max(min(static_cast<int>(cx), width-1), 0);
                        int y0 = max(min(static_cast<int>(cy), height-1), 0);
                        int x1 = min(x0+1, width-1);
                        int y1 = min(y0+1, height-1);
                        if (x0 == w && y0 == h) {
                            gradient += (1-p)*(1-q) * top_diff[offset + row*pooled_width + col];
                            continue;
                        }
                        if (x1 == w && y0 == h) {
                            gradient += (1-p)*q * top_diff[offset + row*pooled_width + col];
                            continue;
                        }
                        if (x0 == w && y1 == h) {
                            gradient += p*(1-q) * top_diff[offset + row*pooled_width + col];
                            continue;
                        }
                        if (x1 == w && y1 == h) {
                            gradient += p * q * top_diff[offset + row*pooled_width + col];
                        }
                    }
                }
            }
            bottom_diff = gradient;
            ''', 'roi_align_2d_bwd')(
            gy[0], bottom_rois.shape[0], self.spatial_scale, channels, height,
            width, self.outh, self.outw, bottom_rois, bottom_diff)

        return bottom_diff, None


def roi_align_2d(x, rois, outh, outw, spatial_scale):
    """Spatial Region of Interest (ROI) align function.

    This function acts similarly to :class:`~functions.RoIPooling2D`, but
    #TODO explain

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~chainer.Variable): Input roi variable. The shape is expected to
            be (n: data size, 5), and each datum is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        outh (int): Height of output image after pooled.
        outw (int): Width of output image after pooled.
        spatial_scale (float): Scale of the roi is resized.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    """
    return ROIAlign2D(outh, outw, spatial_scale)(x, rois)
