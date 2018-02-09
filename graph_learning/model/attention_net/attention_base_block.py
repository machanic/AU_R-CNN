import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from itertools import combinations

linear_init = chainer.initializers.LeCunUniform()




def seq_func(func, x, reconstruct_shape=True):
    """ Change implicitly function's target to ndim=3

    Apply a given function for array of ndim 3,
    shape (batchsize, dimension, sentence_length),
    instead for array of ndim 2.

    """

    batch, units, length = x.shape
    e = F.transpose(x, (0, 2, 1)).reshape(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = F.transpose(e.reshape((batch, length, out_units)), (0, 2, 1))
    assert(e.shape == (batch, out_units, length))
    return e


def sentence_block_embed(embed, x):
    """ Change implicitly embed_id function's target to ndim=2

    Apply embed_id for array of ndim 2,
    shape (batchsize, sentence_length),
    instead for array of ndim 1.

    """

    batch, length = x.shape
    _, units = embed.W.shape
    e = embed(x.reshape((batch * length, )))
    assert(e.shape == (batch * length, units))
    e = F.transpose(F.stack(F.split_axis(e, batch, axis=0), axis=0), (0, 2, 1)) # cut along axis=0 to batch slices
    assert(e.shape == (batch, units, length))
    return e


class LayerNormalizationSentence(L.LayerNormalization):

    """ Position-wise Linear Layer for Sentence Block

    Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length).

    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = seq_func(super(LayerNormalizationSentence, self).__call__, x) # shape = batch, out_units, length
        return y


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
        x = F.expand_dims(x, axis=3) # shape = batchsize, in_channels, sentence_length,1
        # 1d fc along sentence_length in parallel can be simulated by 1x1 convolution
        y = super(ConvolutionSentence, self).__call__(x)
        y = F.squeeze(y, axis=3)
        return y



class MultiHeadAttention(chainer.Chain):

    """ Multi Head Attention Layer for Sentence Blocks

    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.

    """

    def __init__(self, n_units, h=8, dropout=0.1, self_attention=True):
        # n_units is d_k,also as query and key's input dim
        super(MultiHeadAttention, self).__init__()

        with self.init_scope():
            if self_attention:
                self.W_QKV = ConvolutionSentence(
                    n_units, n_units * 3, nobias=True,
                    initialW=linear_init)
            else:
                self.W_Q = ConvolutionSentence(
                    n_units, n_units, nobias=True,
                    initialW=linear_init)
                self.W_KV = ConvolutionSentence(
                    n_units, n_units * 2, nobias=True,
                    initialW=linear_init)

            self.finishing_linear_layer = ConvolutionSentence(
                n_units, n_units, nobias=True,
                initialW=linear_init)
        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        self.dropout = dropout
        self.is_self_attention = self_attention

    def __call__(self, x, z=None, mask=None):
        # pass in x and z shape = (batch, D, T)
        xp = self.xp
        h = self.h
        if self.is_self_attention:
            # expand dimension to 3 times first, then split, thus information of Q and V comes from x in self attenion.
            # self.W_QKV output (batchsize, n_units * 3, sentence_length), then split by axis=1
            Q, K, V = F.split_axis(self.W_QKV(x), 3, axis=1)
        else:
            Q = self.W_Q(x)
            K, V = F.split_axis(self.W_KV(z), 2, axis=1)

        batch, n_units, n_querys = Q.shape  # n_querys is count of query which can also named as "T_q"
        _, _, n_keys = K.shape # n_keys is sentence_length which can also named as "T_k"

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching: h * batch_size
        # all together at once for efficiency
        # batch, n_units, n_querys : split by axis=1 to cut n_heads slice,
        # each slice shape is (batch, n_units//8, n_querys)
        # then concat in axis=0
        batch_Q = F.concat(F.split_axis(Q, h, axis=1), axis=0)
        batch_K = F.concat(F.split_axis(K, h, axis=1), axis=0)
        batch_V = F.concat(F.split_axis(V, h, axis=1), axis=0)
        assert(batch_Q.shape == (batch * h, n_units // h, n_querys))
        assert(batch_K.shape == (batch * h, n_units // h, n_keys))
        assert(batch_V.shape == (batch * h, n_units // h, n_keys))

        # Notice that this formula is different from paper Eqn 1., this time is transpose of Q matrix,
        # This F.matmul actually perform batch_matmul
        batch_A = F.matmul(batch_Q, batch_K, transa=True) \
            * self.scale_score  # shape = batch * h, T_q , T_k，matrix mat along the dimension of n_units//8
        # if mask==False，use -np.inf, so above code:[mask] * h
        if mask is not None:
            mask = xp.concatenate([mask] * h, axis=0)
            batch_A = F.where(mask, batch_A, xp.full(batch_A.shape, -np.inf, 'f'))
        batch_A = F.softmax(batch_A, axis=2)  # axis=2 means attend along n_keys axis. Thus you can softly choose keys
        batch_A = F.where(
            xp.isnan(batch_A.data), xp.zeros(batch_A.shape, 'f'), batch_A) # push nan value to zero
        assert(batch_A.shape == (batch * h, n_querys, n_keys))

        # Calculate Weighted Sum before broad_cast (batch_A stores weights)
        # batch_A shape = (batch * h, 1, n_querys, n_keys); batch_V shape = (batch * h, n_units//8, 1, n_keys)
        batch_A, batch_V = F.broadcast(
            batch_A[:, None], batch_V[:, :, None])
        # batch_C shape = batch * h, n_units//8, n_querys, n_keys, axis=3 means weighted sum along sequence
        batch_C = F.sum(batch_A * batch_V, axis=3) # shape = batch * h, n_units//8, n_querys
        assert(batch_C.shape == (batch * h, n_units // h, n_querys))
        # slice in h piece，then concat along axis=1, shape = (batch, n_units//8 * 8, n_querys), head = 8
        C = F.concat(F.split_axis(batch_C, h, axis=0), axis=1)
        # Notice that there is no n_keys in shape any more, because weighed sum already eliminated this dimension.
        assert(C.shape == (batch, n_units, n_querys))
        C = self.finishing_linear_layer(C)
        return C


class FeedForwardLayer(chainer.Chain):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4  # hidden layer dimension is big than input/output dimension
        with self.init_scope():
            self.W_1 = ConvolutionSentence(n_units, n_inner_units,
                                           initialW=linear_init)
            self.W_2 = ConvolutionSentence(n_inner_units, n_units,
                                           initialW=linear_init)
            # self.act = F.relu
            self.act = F.leaky_relu

    def __call__(self, e):
        e = self.W_1(e)
        e = self.act(e)
        e = self.W_2(e)
        return e


class EncoderLayer(chainer.Chain):

    def __init__(self, n_units, h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h)  # n_units是输入数据的维度
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, e, xx_mask=None):
        sub = self.self_attention(e, e, xx_mask)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_2(e)
        return e  # shape = (batch, n_units, n_querys) alias of (B,D,T)


class DecoderLayer(chainer.Chain):

    def __init__(self, n_units, h=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h)
            self.source_attention = MultiHeadAttention(
                n_units, h, self_attention=False)
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_3 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, target_block, encoded_block, xy_mask, yy_mask):
        # encoded_block is all neighbors' encoded message, shape = (neighbors, batch, unit, seq_length),
        # encoded_block also (neighbors, B, D, T)
        sub = self.self_attention(target_block, target_block, yy_mask)
        e = target_block + F.dropout(sub, self.dropout)
        e = self.ln_1(e)  # shape = (batch, unit, seq_length)

        sub = self.source_attention(e, encoded_block, xy_mask)  # e is query Q

        e = e + F.dropout(sub, self.dropout)
        e = self.ln_2(e)
        sub = self.feed_forward(e)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_3(e)
        return e


class Encoder(chainer.Chain):

    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = EncoderLayer(n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, xx_mask=None):
        for name in self.layer_names:
            e = getattr(self, name)(e, xx_mask=None)
        return e


class Decoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Decoder, self).__init__()
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = DecoderLayer(n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, source, xy_mask, yy_mask):
        for name in self.layer_names:
            e = getattr(self, name)(e, source, xy_mask, yy_mask)
        return e






class AttentionBlock(chainer.Chain):

    def __init__(self, n_layers, n_units, out_size, h=8, dropout=0.5, max_length=1500):
        super(AttentionBlock, self).__init__()
        self.layer_names = []
        self.n_units = n_units
        self.scale_emb = self.n_units ** 0.5
        self.position_encoding_block = None
        self.initialize_position_encoding(max_length, n_units)
        with self.init_scope():
            self.encoder = Encoder(n_layers, n_units, h, dropout)
            self.final_linear = ConvolutionSentence(in_channels=n_units, out_channels=out_size)

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
        signal = xp.reshape(signal, [1, length, channels])
        self.position_encoding_block = xp.transpose(signal, (0, 2, 1)) # shape = [1, channels(n_units), length]



    def __call__(self, x, mask=None):
        # x is shape = (batch, D, T)
        batch, unit, length = x.shape
        x += self.xp.array(self.position_encoding_block[:,:,:length]) # FIXME is it right to use it
        h = self.encoder(x, mask)  # self attention
        h = self.final_linear(h)
        return h  # shape = (batch, out_size, T)


