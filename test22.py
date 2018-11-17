import math
import numpy as np
from chainer.cuda import cupy
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import initializers
from chainer import Variable
import pdb

import sys
sys.path.append('./')
import link_binary_linear as BL
import bst
import link_binary_conv2d as BC
import link_integer_conv2d as IC
import link_residual_block as RB
import link_dense_block as DB
import link_dorefa_conv2d as DC
import link_dorefa_linear as DL
import dorefa_activation as da

from function_binary_conv2d import func_convolution_2d
from function_integer_conv2d import func_convolution_2d

# for debuging of the batch normalization functions
import link_batch_normalization as LBN

class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__(

            conv0=L.Convolution2D(3,16,7, stride=2, pad=3, nobias=True),
            b_conv0=L.BatchNormalization(16),
            block0=DB.BlockStack(6,16,88,k=12,kernel=(1,3),stride=(1,1),pad=(0,1),nobias=False), 
            b_block0=L.BatchNormalization(88),
            conv1=DC.Convolution2D(88,88,1, stride=1, pad=0, nobias=True),
            b_conv1=L.BatchNormalization(88),
            block1=DB.BlockStack(12,88,232,k=12,kernel=(1,3),stride=(1,1),pad=(0,1),nobias=False), 
            b_block1=L.BatchNormalization(232),
            conv2=DC.Convolution2D(232,232,1, stride=1, pad=0, nobias=True),
            b_conv2=L.BatchNormalization(232),
            block2=DB.BlockStack(24,232,520,k=12,kernel=(1,3),stride=(1,1),pad=(0,1),nobias=False), 
            b_block2=L.BatchNormalization(520),
            conv3=DC.Convolution2D(520,520,1, stride=1, pad=0, nobias=True),
            b_conv3=L.BatchNormalization(520),
            fc0=DL.DorefaLinear(2080,3),
            b_dense0=L.BatchNormalization(3)
        )

    def __call__(self, x, train,batch_size):
        h = self.b_conv0(F.elu(self.conv0(x)))
        h = F.max_pooling_2d(h, 3, 2)
        h = self.block0(h, train)
        h = self.b_conv1(da.dst(self.conv1(h)))
        h = F.max_pooling_2d(h, 3, 2)
        h = self.block1(h, train)
        h = self.b_conv2(da.dst(self.conv2(h)))
        h = F.max_pooling_2d(h, 3, 2)
        h = self.block2(h, train)
        h = self.b_conv3(da.dst(self.conv3(h)))
        h = F.max_pooling_2d(h, 2, 2)
        pdb.set_trace()
        h = Variable(cupy.reshape(h.data,[batch_size,-1]),volatile = not train)
        h = self.b_dense0(self.fc0(h))
        return h