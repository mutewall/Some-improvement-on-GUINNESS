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

            conv0=DC.Convolution2D(3,64,7, stride=2, pad=3, nobias=True),
            b_conv0=L.BatchNormalization(64),
            block0=RB.BlockStack(3,64,64,decre_ratio=4, kernel=(1,3,1), stride=(1,1,1), pad=(0,1,0),nobias=True), 
            b_block0=L.BatchNormalization(64),
            conv1=BC.Convolution2D(64,128,3, stride=2, pad=1, nobias=True),
            b_conv1=L.BatchNormalization(128),
            block1=RB.BlockStack(3,128,128,decre_ratio=4, kernel=(1,3,1), stride=(1,1,1), pad=(0,1,0),nobias=True), 
            b_block1=L.BatchNormalization(128),
            conv2=BC.Convolution2D(128,256,3, stride=2, pad=1, nobias=True),
            b_conv2=L.BatchNormalization(256),
            block2=RB.BlockStack(3,256,256,decre_ratio=4, kernel=(1,3,1), stride=(1,1,1), pad=(0,1,0),nobias=True), 
            b_block2=L.BatchNormalization(256),
            conv3=BC.Convolution2D(256,512,3, stride=2, pad=1, nobias=True),
            b_conv3=L.BatchNormalization(512),
            block3=RB.BlockStack(3,512,512,decre_ratio=4, kernel=(1,3,1), stride=(1,1,1), pad=(0,1,0),nobias=True), 
            b_block3=L.BatchNormalization(512),
            conv4=BC.Convolution2D(512,1024,3, stride=2, pad=1, nobias=True),
            b_conv4=L.BatchNormalization(1024),
            fc0=BL.BinaryLinear(1024,3),
            b_dense0=L.BatchNormalization(3)
        )

    def __call__(self, x, train,batch_size):
        h = self.b_conv0(da.dst(self.conv0(x)))
        h = self.block0(h, train)
        h = bst.bst(self.b_conv1(self.conv1(h)))
        h = self.block1(h, train)
        h = bst.bst(self.b_conv2(self.conv2(h)))
        h = self.block2(h, train)
        h = bst.bst(self.b_conv3(self.conv3(h)))
        h = self.block3(h, train)
        h = bst.bst(self.b_conv4(self.conv4(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.average_pooling_2d(h, 2, 2, 1)
        h = Variable(cupy.reshape(h.data,[batch_size,-1]),volatile = not train)
        h = self.b_dense0(self.fc0(h))
        return h