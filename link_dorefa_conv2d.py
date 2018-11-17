import math

import function_dorefa_conv2d
from chainer import initializers
from chainer import link

import numpy


class Convolution2D(link.Link):


    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=False,
                 initialW=None, initial_bias=None):
        super(Convolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.use_cudnn = use_cudnn
        self.out_channels = out_channels
        #self.initialW = initializers.LeCunUniform()
        self.initialW = initializers.Normal()
        self.wscale = wscale

        if in_channels is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_channels)

        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        #self.add_param('W', W_shape)
        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        initializers.init_weight(self.W.data, self.initialW,
                                 scale=math.sqrt(self.wscale))

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_channels)
            if initial_bias is None:
                initial_bias = bias
            initializers.init_weight(self.b.data, initial_bias)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        self.add_param('W', W_shape)
        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        initializers.init_weight(self.W.data, self.initialW,
                                 scale=math.sqrt(self.wscale))

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        return function_dorefa_conv2d.func_convolution_2d(x, self.W, self.b, self.stride, self.pad, self.use_cudnn)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
