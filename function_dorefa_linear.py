import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

E_w = 4
E_a = 2**8-1
E_g = 2**16-1

def _kern_x():
    return cuda.elementwise(
        'T x, T E', 'T y',
        'y = x/E',
        'quantize_x')
def _kern_w():
    return cuda.elementwise(
        'T w, T E', 'T y',
        'y = 2*w/E - 1',
        'quantize_w')

def quantize(x, E):
    return numpy.round(x*E)/E 

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class DorefaLinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_cpu(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        
        Wq = quantize(numpy.clip(W*0.5+0.5, 0, 1).astype(numpy.float32, copy=False), E_w).astype(numpy.float32, copy=False)
        Wq = 2*Wq - 1

        #Xq = quantize(x,E_a).astype(x.dtype, copy=False)

        y = x.dot(Wq.T)

        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,        

    def forward_gpu(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        
        W_temp = cuda.cupy.clip(W*0.5+0.5, 0, 1)
        W_temp = cuda.cupy.rint(W_temp*E_w)
        Wq = _kern_w()(W_temp, E_w)
        assert W.shape == Wq.shape

        #x = cuda.cupy.rint(x*E_a)
        #Xq = _kern_x()(x, E_a)
        
        y = x.dot(Wq.T)

        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,        


    def backward_cpu(self,inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        Wb = quantize(W, E_w)
        gy = grad_outputs[0]
        """
        coef = numpy.max(gy, axis=tuple([i for i in xrange(1,gy.ndim)])).astype(numpy.float32)
        coef = _as_mat(coef)
        gy = _as_mat(gy)
        coef_invert = 0.5*coef**(-1)
        gy = gy*coef_invert+0.5
        gy = quantize(gy, E_g)
        gy = 2*coef*(gy-0.5)
        gy = gy.reshape(grad_outputs[0].shape)
        """
        
        gx = gy.dot(Wb).reshape(inputs[0].shape)
        gW = gy.T.dot(x)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW
    
    def backward_gpu(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]

        W = cuda.cupy.clip(W*0.5+0.5, 0, 1)
        W = cuda.cupy.rint(W*E_w)
        Wq = _kern_w()(W, E_w)
        gy = grad_outputs[0]
        """
        coef = cuda.cupy.max(gy, axis=tuple([i for i in xrange(1,gy.ndim)])).astype(cuda.cupy.float32)
        coef = _as_mat(coef)
        gy = _as_mat(gy)
        coef_invert = 0.5*coef**(-1)
        gy = gy*coef_invert+0.5
        gy = cuda.cupy.rint(gy*E_g)
        gy = _kern_x()(gy,E_g)
        gy = 2*coef*(gy-0.5)
        gy = gy.reshape(grad_outputs[0].shape)
        """
        gx = gy.dot(Wq).reshape(inputs[0].shape)
        gW = gy.T.dot(x)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW

def dorefa_linear(x, W, b=None):
    if b is None:
        return DorefaLinearFunction()(x, W)
    else:
        return DorefaLinearFunction()(x, W, b)