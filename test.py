import math
import numpy as np
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

            conv0=L.Convolution2D(3,64,7, stride=2, pad=3, nobias=True),
            b_conv0=L.BatchNormalization(64),
            block0=RB.BlockStack(3,64,64,decre_ratio=4, kernel=(1,3,1), stride=(1,2,1), pad=(0,0,0),nobias=True), 
            b_block0=L.BatchNormalization(64),
            conv1=IC.Convolution2D(64,128,3, stride=1, pad=1, nobias=True),
            b_conv1=L.BatchNormalization(128),
            block1=RB.BlockStack(3,128,128,decre_ratio=4, kernel=(1,3,1), stride=(1,2,1), pad=(0,0,0),nobias=True), 
            b_block1=L.BatchNormalization(128),
            conv2=IC.Convolution2D(128,256,3, stride=1, pad=1, nobias=True),
            b_conv2=L.BatchNormalization(256),
            block2=RB.BlockStack(3,256,256,decre_ratio=4, kernel=(1,3,1), stride=(1,2,1), pad=(0,0,0),nobias=True), 
            b_block2=L.BatchNormalization(256),
            conv3=IC.Convolution2D(256,512,3, stride=1, pad=1, nobias=True),
            b_conv3=L.BatchNormalization(512),
            block3=RB.BlockStack(3,512,512,decre_ratio=4, kernel=(1,3,1), stride=(1,2,1), pad=(0,0,0),nobias=True), 
            b_block3=L.BatchNormalization(512),
            conv4=IC.Convolution2D(512,1024,3, stride=1, pad=1, nobias=True),
            b_conv4=L.BatchNormalization(1024),
            fc0=BL.BinaryLinear(4096,10),
            b_dense0=L.BatchNormalization(10)
        )

    def __call__(self, x, train,batch_size):
        h = self.b_conv0(F.elu(self.conv0(x)))
        h = self.block0(h, train)
        h = self.b_conv1(F.relu(self.conv1(h)))
        h = self.block1(h, train)
        h = self.b_conv2(F.relu(self.conv2(h)))
        h = self.block2(h, train)
        h = self.b_conv3(F.relu(self.conv3(h)))
        h = self.block3(h, train)
        h = self.b_conv4(F.relu(self.conv4(h)))
        h = F.max_pooling_2d(h, 3, 2, 1)
        h = F.average_pooling_2d(h, 2, 2)
        h = Variable(h.data.reshape([batch_size,-1]),volatile=not train)
        h = self.b_dense0(self.fc0(h))
        return h




"""

import math
import numpy as np
import six
import chainer
import argparse
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import initializers
from chainer import Variable
from chainer import iterators
from chainer import training
from chainer import optimizers
from chainer.training import extensions

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

call_link = IC
call_acti = bst.bst
call = RB

class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__(

            conv0=L.Convolution2D(3,64,7, stride=1, pad=3, nobias=False),
            b_conv0=L.BatchNormalization(64),
            block0=call.BlockStack(3,64,64,decre_ratio=4, kernel=(1,3,1), stride=(1,2,1), pad=(0,1,0),nobias=True), 
            b_block0=L.BatchNormalization(64),
            conv1=call_link.Convolution2D(64,128,3, stride=2, pad=1, nobias=False),
            b_conv1=L.BatchNormalization(128),
            block1=call.BlockStack(3,128,128,decre_ratio=4, kernel=(1,3,1), stride=(1,2,1), pad=(0,1,0),nobias=True), 
            b_block1=L.BatchNormalization(128),
            conv2=call_link.Convolution2D(128,256,3, stride=2, pad=1, nobias=False),
            b_conv2=L.BatchNormalization(256),
            block2=call.BlockStack(3,256,256,decre_ratio=4, kernel=(1,3,1), stride=(1,2,1), pad=(0,1,0),nobias=True), 
            b_block2=L.BatchNormalization(256),
            conv3=call_link.Convolution2D(256,512,3, stride=2, pad=1, nobias=False),
            b_conv3=L.BatchNormalization(512),
            block3=call.BlockStack(3,512,512,decre_ratio=4, kernel=(1,3,1), stride=(1,2,1), pad=(0,1,0),nobias=True), 
            b_block3=L.BatchNormalization(512),
            conv4=call_link.Convolution2D(512,1024,3, stride=2, pad=1, nobias=False),
            b_conv4=L.BatchNormalization(1024),
            fc0=L.Linear(1024,10),#modified
            b_dense0=L.BatchNormalization(10),
            #fc1=L.Linear(1050,5),
            #b_dense1=L.BatchNormalization(5)
        )

    def __call__(self, x, train):
        #pdb.set_trace()
        h = self.b_conv0(F.elu(self.conv0(x)))
        #h = F.max_pooling_2d(h, 2, 2)
        #quit
		# pdb.set_trace()
        h = self.block0(h, train)
        h = self.b_conv1(call_acti(self.conv1(h)))
        #h = F.max_pooling_2d(h, 2, 2)
        h = self.block1(h, train)
        h = self.b_conv2(call_acti(self.conv2(h)))
        #h = F.max_pooling_2d(h, 2, 2)
        h = self.block2(h, train)
        h = self.b_conv3(call_acti(self.conv3(h)))
        #h = F.max_pooling_2d(h, 2, 2)
        h = self.block3(h, train)
        h = self.b_conv4(call_acti(self.conv4(h)))
        h = F.max_pooling_2d(h, h.shape[2:])
        h = self.fc0(h)
        #h = self.b_dense0(call_acti(self.fc0(h)))
        #pdb.set_trace()
        #h = self.b_dense1(self.fc1(h))
        #pdb.set_trace()
        return h

"""
def main():
    parser = argparse.ArgumentParser(description='Chainer test: DoRes')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = L.Classifier(CNN())
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_cifar10()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
"""