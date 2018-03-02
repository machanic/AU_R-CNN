import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from enum import Enum

linear_init = chainer.initializers.LeCunUniform()


def seq_func(func, x, reconstruct_shape=True):
    """ Change implicitly function's target to ndim=3

    Apply a given function for array of ndim 3,
    shape (batchsize, dimension, sentence_length),
    instead for array of ndim 2.

    """
    batch, length, units = x.shape
    e = F.reshape(x, shape=(batch * length, units))
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = e.reshape(batch, length, out_units)
    return e

class LayerNormalizationSentence(L.LayerNormalization):

    """ Position-wise Linear Layer for Sentence Block

    Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length).

    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = seq_func(super(LayerNormalizationSentence, self).__call__, x) # shape = batch, length, out_units
        return y


class ScaledDotProductAttention(chainer.Chain):
    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.attn_dropout = attn_dropout

    def __call__(self, q, k, v, attn_mask=None):
        # q k v shape = (n_head* batch) x len_q/len_k x d_k or d_v
        attn = F.matmul(q, F.transpose(k, axes=(0,2,1))) / self.temper
        # # (n_head*batch) of matrix multiply (len_q x d_k) x (d_k x len_k) =  (n_head*batch) x len_q x len_k

        if attn_mask is not None:
            assert attn_mask.shape == attn.shape, \
                'Attention mask shape {0} mismatch ' \
                'with Attention logit tensor shape ' \
                '{1}.'.format(attn_mask.shape, attn.shape)
            if hasattr(attn_mask, "data"):
                attn_mask.data = attn_mask.data.astype(bool)
            attn = F.where(attn_mask, self.xp.full(attn.shape, -np.inf, 'f'), attn)
        attn = F.softmax(attn, axis=2)  # (n_head*batch) x len_q x len_k
        attn = F.dropout(attn, ratio=self.attn_dropout)
        output = F.matmul(attn, v) #  (n_head*batch) matrix of (len_q x len_k) x (len_v x d_v) = (n_head*batch) x len_q x d_v
        # 因为d_k == d_q，所以 output = (n_head*batch) x len_q x d_v
        return output, attn

class MultiHeadAttention(chainer.Chain):

    def __init__(self, n_heads, d_model, d_k=64, d_v=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout_ratio = dropout

        with self.init_scope():
            self.w_qs = chainer.Parameter(linear_init, shape=(n_heads, d_model, d_k))
            self.w_ks = chainer.Parameter(linear_init, shape=(n_heads, d_model, d_k))
            self.w_vs = chainer.Parameter(linear_init, shape=(n_heads, d_model, d_v))

            self.attention = ScaledDotProductAttention(d_model)
            self.layer_norm = LayerNormalizationSentence(d_model,eps=1e-6)
            self.proj = L.Linear(n_heads * d_v, d_model) # note that typical case d_v = d_model // n_heads


    def __call__(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_heads
        batch_size, len_q, d_model = q.shape
        batch_size, len_k, d_model = k.shape
        batch_size, len_v, d_model = v.shape
        residual = q
        # treat as a (n_head) size batch, shape = (heads x batch), number_words, d_model; then (heads, (batch x len_q), d_model)
        q_s = F.tile(q, reps=(n_head, 1, 1)).reshape(n_head, -1, d_model)  # n_head x (batch_size*len_q) x d_model
        k_s = F.tile(k, reps=(n_head, 1, 1)).reshape(n_head, -1, d_model)  # n_head x (batch_size*len_k) x d_model
        v_s = F.tile(v, reps=(n_head, 1, 1)).reshape(n_head, -1, d_model)  # n_head x (batch_size*len_v) x d_model

        # (n_head) batch matrix multiply of  ((batch * len_q) x d_model) x (d_model, d_k) = (batch * len_q) x d_k
        # treat the result as a (n_head * mb_size) size batch
        q_s = F.matmul(q_s, self.w_qs).reshape(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = F.matmul(k_s, self.w_ks).reshape(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = F.matmul(v_s, self.w_vs).reshape(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # outputs size = (n_head * mb_size) x len_q x d_v, attns size = (n_head*mb_size) x len_q x len_k
        if attn_mask is not None:
            attn_mask = F.tile(attn_mask, reps=(n_head, 1, 1))
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask) #  (n_head*batch) x len_q x d_v
        outputs = F.concat(F.split_axis(outputs, n_head, axis=0), axis=2) # = batch_size, len_q, (n_head*d_v)
        outputs = F.reshape(outputs, shape=(batch_size * len_q, n_head * d_v))
        # project back to residual size
        outputs = self.proj(outputs)
        outputs = F.dropout(outputs, self.dropout_ratio)
        outputs = F.reshape(outputs, shape=(batch_size, len_q, d_model))
        return self.layer_norm(outputs + residual)



class ConvolutionSentence(L.Convolution2D):

    """ Position-wise Linear Layer for Sentence Block

    Position-wise linear layer for array of shape
    (batchsize, dimension, sentence_length)
    can be implemented a convolution layer.

    """

    def __init__(self, in_channels, out_channels,
                 ksize=1, stride=1, pad=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(ConvolutionSentence, self).__init__(
            in_channels, out_channels,
            ksize, stride, pad, nobias,
            initialW, initial_bias)

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vector block. Its shape is
                (batchsize, in_channels, sentence_length).

        Returns:
            ~chainer.Variable: Output of the linear layer. Its shape is
                (batchsize, out_channels, sentence_length).

        """
        x = F.expand_dims(x, axis=3) # shape = (batch_size, in_channels, sentence_length, 1)
        # 1d fc along sentence_length in parallel can be simulated by 1x1 convolution
        y = super(ConvolutionSentence, self).__call__(x)
        y = F.squeeze(y, axis=3)
        return y

class PositionFFNType(Enum):
    regular = 1
    nstep_lstm = 2

class PositionwiseFeedForwardLayer(chainer.Chain):
    def __init__(self, n_layers, in_size, out_size=None, n_inner_units=1024, dropout=0.1, forward_type=PositionFFNType.nstep_lstm):
        super(PositionwiseFeedForwardLayer, self).__init__()
        self.dropout_ratio = dropout
        self.forward_type = forward_type
        assert n_layers >= 2
        if out_size is None:
            out_size = in_size
        # n_inner_units = in_size * 4  # hidden layer dimension is big than input/output dimension
        with self.init_scope():
            self.layer_name = []
            if n_layers == 2:
                self.w0 = ConvolutionSentence(in_size, n_inner_units,
                                              initialW=linear_init)
                self.w1 = ConvolutionSentence(n_inner_units, out_size,
                                              initialW=linear_init)
                self.layer_name.append("w0")
                self.layer_name.append("w1")
            elif n_layers > 2:
                self.w0 = ConvolutionSentence(in_size, n_inner_units,
                                               initialW=linear_init)
                for i in range(1, n_layers-1):
                    setattr(self, "w{}".format(i), ConvolutionSentence(n_inner_units, n_inner_units,
                                          initialW=linear_init))

                setattr(self, "w{}".format(n_layers-1),ConvolutionSentence(n_inner_units, out_size,
                                          initialW=linear_init))

            self.layer_norm = LayerNormalizationSentence(out_size)
            self.act = F.leaky_relu

    def __call__(self, x):
        if self.forward_type == PositionFFNType.nstep_lstm:
            return self.forward_nstep_lstm(x)
        elif self.forward_type == PositionFFNType.regular:
            return self.forward_regular(x)

    def forward_regular(self, x):  # shape of x = batch, T, d_model
        residual = x
        x = F.transpose(x, axes=(0,2,1))
        for conv_layer in self.layer_name:
            x = self.act(getattr(self, conv_layer)(x))
        output = F.transpose(x, axes=(0,2,1))
        output = F.dropout(output, self.dropout_ratio)
        return self.layer_norm(output + residual)


    def forward_nstep_lstm(self, es):
        # e shape = list of (sentence_length, in_channels)
        out_es = []
        for e in es:
            e = F.transpose(e, axes=(1, 0))  # D, T
            e = F.expand_dims(e, axis=0) # 1,D,T
            for conv_layer in self.layer_name:
                e = self.act(getattr(self, conv_layer)(e))
            e = F.transpose(e, axes=(0, 2, 1))[0]  # return B, T, D, then [0] = T,D
            e = F.dropout(e, self.dropout_ratio)
            out_es.append(e)
        out_es = F.stack(out_es) # B,T,D
        return [F.squeeze(e) for e in F.split_axis(self.layer_norm(out_es), out_es.shape[0], axis=0, force_tuple=True)]  # return list of (T,D)

class EncoderLayer(chainer.Chain):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.slf_attn = MultiHeadAttention(n_heads=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)

            # self.pos_ffn = PositionwiseFeedForwardLayer(n_layers=1, in_size=d_model, n_inner_units=d_inner_hid,
            #                                             dropout=dropout, forward_type=PositionFFNType.regular)

    def __call__(self, enc_input, slf_attn_mask=None):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=slf_attn_mask) # # enc_output shape = mb_size x len_q x d_model
        # enc_output = self.pos_ffn(enc_output)
        return enc_output # shape = batch x len_q x d_model

class Encoder(chainer.Chain):

    def __init__(self, n_layer=6, n_head=8,d_model=2048, d_inner_hid=1024, d_k=256, d_v=256,dropout=0.1):
        super(Encoder, self).__init__()
        self.layer_names = []
        assert d_k == d_model//n_head
        assert d_v == d_model//n_head
        with self.init_scope():
            for i in range(1, n_layer+1):
                name = "l{}".format(i)
                layer = EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
                self.add_link(name, layer)
                self.layer_names.append(name)

    def __call__(self, e, xx_mask):
        for name in self.layer_names:
            e = getattr(self, name)(e, slf_attn_mask=xx_mask)
        return e

class AttentionBlock(chainer.Chain):
    def __init__(self, n_layers, d_model, out_size, d_inner_hid=1024, n_head=4, dropout=0.1, max_length=1500):
        super(AttentionBlock, self).__init__()
        self.layer_names = []
        self.out_size = out_size
        self.d_model = d_model
        self.position_encoding_block = self.initialize_position_encoding(max_length, d_model)
        d_k = d_model // n_head
        d_v = d_model // n_head
        with self.init_scope():
            self.encoder = Encoder(n_layers, n_head=n_head, d_model=d_model,  d_inner_hid=d_inner_hid, d_k=d_k, d_v=d_v,
                                   dropout=dropout)
            # 本来是ConvolutionSentence，修改fc层权宜之计，因为显存不够用
            self.final_linear = L.Linear(d_model, out_size)

    def initialize_position_encoding(self, length, n_units):
        xp = self.xp
        # Implementation in the Google tensor2tensor repo
        channels = n_units
        position = xp.arange(length, dtype='f')
        num_timescales = channels // 2
        log_timescale_increment = (
            xp.log(10000. / 1.) / (float(num_timescales) - 1))
        inv_timescales = 1. * xp.exp(
            xp.arange(num_timescales).astype('f') * - log_timescale_increment)
        scaled_time = \
            xp.expand_dims(position, 1) * \
            xp.expand_dims(inv_timescales, 0)
        signal = xp.concatenate(
            [xp.sin(scaled_time), xp.cos(scaled_time)], axis=1)
        signal = xp.reshape(signal, [1, length, channels]) # shape = [1, length, channels(n_units)]
        return signal

    def __call__(self, x_lst, mask=None):
        # x is shape = (batch, T, D)
        x = F.stack(x_lst)
        batch, length, unit = x.shape
        x += self.xp.array(self.position_encoding_block[:, :length, :])
        h = self.encoder(x, mask)  # self attention shape= batch x len_q x d_model
        batch, len_q, d_model = h.shape
        h = F.reshape(h, (batch*len_q, d_model))
        h = self.final_linear(h)  # shape  = B, out_size, len_q
        h = F.reshape(h, (batch, len_q, self.out_size))
        # shape = B, len_q, out_size , then convert to [len_q, out_size] that is list of T,D
        # return [F.squeeze(e) for e in F.split_axis(F.transpose(h, axes=(0, 2, 1)), 1, axis=0, force_tuple=True)]
        return [F.squeeze(e) for e in F.split_axis(h, 1, axis=0, force_tuple=True)]