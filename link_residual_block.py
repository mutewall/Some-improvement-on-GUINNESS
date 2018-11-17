import chainer
import chainer.links as L
import link_integer_conv2d as IC
import link_binary_conv2d as BC
import link_dorefa_conv2d as DC
import link_batch_normalization as BN
from chainer import functions as F
from chainer import cuda
from chainer import Variable
import bst
import dorefa_activation as da
#import pdb


call_link = BC
call_acti = bst.bst

class BlockStack(chainer.ChainList):
	def __init__(self, n_stack, in_channels, out_channels, decre_ratio, kernel, stride, pad, nobias=True, use_cudnn=True, initialW=None,initial_bias=None, proj=False):
		super(BlockStack, self).__init__()
		w = chainer.initializers.HeNormal()
		mid_channels = in_channels/decre_ratio

		if in_channels!= out_channels or decre_ratio != 1:
			proj = True

		#with cuda.get_device_from_id(0):
		for _ in xrange(n_stack):
			self.add_link(BottleNeck(in_channels, mid_channels, out_channels, kernel, stride, pad, proj, nobias))
	
	def __call__(self, x, train):
		for f in self.children():
			x = f(x, train)
		return x


class BottleNeck(chainer.Chain):
	def __init__(self, in_channels, mid_channels, out_channels, kernel, stride, pad, proj, nobias, use_cudnn=True, initialW=None, initial_bias=None):
		w = chainer.initializers.Normal()
		k1,k2,k3 = kernel
		s1,s2,s3 = stride
		p1,p2,p3 = pad
		super(BottleNeck, self).__init__(
            conv1 = call_link.Convolution2D(in_channels, mid_channels, k1, s1, p1, nobias=nobias, initialW=w),
		    conv2 = call_link.Convolution2D(mid_channels, mid_channels, k2, s2, p2, nobias=nobias, initialW=w),
		    conv3 = call_link.Convolution2D(mid_channels, out_channels, k3, s3, p3, nobias=nobias, initialW=w),
		    bn1 = L.BatchNormalization(mid_channels),
		    bn2 = L.BatchNormalization(mid_channels),
		    bn3 = L.BatchNormalization(out_channels)
        )
		if proj :
			self.add_link('shortcut_conv' ,call_link.Convolution2D(in_channels, out_channels, k2, s2, p2, nobias=nobias, initialW=w))
			self.add_link('shortcut_bn',L.BatchNormalization(out_channels))
		self.proj = proj
	def __call__(self, x, train):
		h = call_acti(self.bn1(self.conv1(x)))
		h = call_acti(self.bn2(self.conv2(h)))
		h = self.bn3(self.conv3(h))
		if self.proj:
			x = self.shortcut_bn(self.shortcut_conv(x))
		return call_acti(h + x)