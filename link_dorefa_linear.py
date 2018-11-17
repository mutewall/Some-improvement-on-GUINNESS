import numpy

from chainer import link
import function_dorefa_linear

class DorefaLinear(link.Link):
    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(DorefaLinear, self).__init__(W=(out_size, in_size))
        if initialW is None:
            #initialW = numpy.random.normal(
                #0, wscale * numpy.sqrt(1. / in_size), (out_size, in_size))
           initialW = numpy.random.normal(
                0, wscale * numpy.sqrt(1. / in_size), (out_size, in_size))        
        self.W.data[...] = initialW

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_size)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        return function_dorefa_linear.dorefa_linear(x, self.W, self.b)
