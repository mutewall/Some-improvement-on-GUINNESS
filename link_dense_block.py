import chainer
from chainer import functions as F
import chainer.links as L
import link_binary_conv2d as BC
import link_dorefa_conv2d as DC
from chainer import cuda
from chainer import Variable
import bst
import dorefa_activation as da
import link_integer_conv2d as IC

call_link = DC
call_acti = da.dst

class BlockStack(chainer.ChainList):
	def __init__(self, n_stack, in_channels, out_channels, k, kernel, stride, pad, 
		nobias=True, use_cudnn=True, initalW=None, initial_bias=None):
		super(BlockStack, self).__init__()

		self.n_stack = n_stack
		for l in xrange(self.n_stack):
			self.add_link(BottleNeck((k*l+in_channels), k, kernel, 
				stride, pad, nobias))
		
		self.list = []

	def __call__(self, x, train):
		i = 1
		self.list.append(x)
		for f in self.children():
			temp_array = [list_x[:].data for list_x in self.list[:i]]
			temp_array = cuda.cupy.concatenate(tuple(temp_array),axis=1)
			x = Variable(cuda.cupy.asarray(temp_array),volatile=not train)
			x = f(x, train)
			self.list.append(x)
			i += 1
		if i != (self.n_stack + 1): print "ERROR!"
		else:
			temp_array = [list_x[:].data for list_x in self.list[:i]]
			temp_array = cuda.cupy.concatenate(tuple(temp_array),axis=1)
			x = Variable(cuda.cupy.asarray(temp_array),volatile=not train)
			self.list = []
			return x



class BottleNeck(chainer.Chain):
	def __init__(self, in_channels, out_channels, kernel, stride, pad, nobias,
		use_cudnn=True, initalW=None, initial_bias=None):
		
		w = chainer.initializers.Normal()
		k1,k2 = kernel
		s1,s2 = stride
		p1,p2 = pad

		super(BottleNeck,self).__init__(
			conv1=call_link.Convolution2D(in_channels,out_channels, k1, s1, p1, nobias=nobias, initialW=w),
			conv2=call_link.Convolution2D(out_channels, out_channels, k2, s2, p2, nobias=nobias,initialW=w),
			b1=L.BatchNormalization(in_channels),
			b2=L.BatchNormalization(out_channels),
		)
		

		#how to deal with mid_channels?			
		

	def __call__(self, x, train):

		h = self.conv1(call_acti(self.b1(x)))
		h = self.conv2(call_acti(self.b2(h)))

		return h