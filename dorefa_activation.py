import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

"""
class DST(function.Function):

    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = (numpy.tanh(x)+1)/2
        return y,

    def forward_gpu(self, x):
        y = (cuda.cupy.tanh(x)+1)/2
        return y,

    def backward_cpu(self, x, gy):
        x = numpy.cosh(x[0])**2*(-1)
        gx = x*gy
        return gx,

    def backward_gpu(self, x, gy):
        x = cuda.cupy.cosh(x[0])**2*(-1)
        gx = x*gy
        return gx,


def dst(x):
    return DST()(x)
"""


class DST(function.Function):

    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0]
        y = numpy.clip(y, 0, 1).astype(numpy.float32, copy=False)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x', 'T y',
            'y = x >0?(x>1 ? 1 : x ):0', 'bst_fwd')(
                x[0])
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = x[0] >= 1
        gx[zero_indices] = 0
        zero_indices = x[0] < 0
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = x >0?(x>1 ? 0 : gy ):0', 'bst_bwd')(
                x[0], gy[0])
        return gx,


def dst(x):
    return DST()(x)


"""
class DST(function.Function):

    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0]
        y = numpy.clip(y, -1, 1).astype(numpy.float32, copy=False)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x', 'T y',
            'y = x<1 ? (x<-1 ? 1 : x) : 1', 'bst_fwd')(
                x[0])
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = x[0] >= 1
        gx[zero_indices] = 0
        zero_indices = x[0] < 0
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = x >0?(x>1 ? 0 : gy ):0', 'bst_bwd')(
                x[0], gy[0])
        return gx,


def dst(x):
    return DST()(x)
"""