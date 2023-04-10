################################################################
# idl_ch2_mlp.py
#
# Programs from Eugene Charniak's Intro to DL, Ch 2
#
################################################################

from tf1 import tf
from input_data import mnist
from mnist_vis import *

import sys

import math
import numpy as np
import matplotlib.pyplot as plt

################################################################
# Fig 2.2
################################################################
def fig2_2( numBatches=1000 ):
    batchSz=100
    numPixels = 784
    numClasses = 10

    print("Fig 2.2 (Single linear layer)")

    W = tf.Variable(tf.random_normal([numPixels, numClasses],stddev=.1))
    b = tf.Variable(tf.random_normal([numClasses], stddev=.1))

    img = tf.placeholder(tf.float32, [batchSz, numPixels])
    ans = tf.placeholder(tf.float32, [batchSz, numClasses])

    prbs = tf.nn.softmax(tf.matmul(img,W)+b)
    xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(xEnt)
    numCorrect = tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
    accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #-----------------------------------------------------
    for i in range(numBatches):
        imgs, anss = mnist.train.next_batch(batchSz)
        sess.run(train, feed_dict={img: imgs, ans: anss})

    print("  Training complete.")

    sumAcc=0
    for i in range(numBatches):
        imgs, anss = mnist.test.next_batch(batchSz)
        sumAcc+=sess.run(accuracy, feed_dict={img: imgs, ans: anss})
    print("  Test Accuracy: %r" % (sumAcc / numBatches))

    sess.close()


################################################################
# Fig 2.9
################################################################
def fig2_9( numBatches = 1000 ):
    batchSz=100
    numPixels = 784
    numClasses = 10

    print("Fig 2.9 (Multi-layer w. Relu)")

    # New
    U = tf.Variable(tf.random_normal([numPixels, numPixels],stddev=.1))
    bU = tf.Variable(tf.random_normal([numPixels], stddev=.1))

    V = tf.Variable(tf.random_normal([numPixels, numClasses], stddev=.1))
    bV = tf.Variable(tf.random_normal([numClasses], stddev=.1))

    img = tf.placeholder(tf.float32, [batchSz, numPixels])
    ans = tf.placeholder(tf.float32, [batchSz, numClasses])

    L1Output = tf.matmul(img,U) + bU
    L1Output = tf.nn.relu(L1Output)
    prbs = tf.nn.softmax(tf.matmul(L1Output,V)+bV)

    # Below is same as Fig 2.2
    xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(xEnt)
    numCorrect = tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
    accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #-----------------------------------------------------
    for i in range(numBatches):
        imgs, anss = mnist.train.next_batch(batchSz)
        sess.run(train, feed_dict={img: imgs, ans: anss})


    print("  Training complete.")

    sumAcc=0
    for i in range(numBatches):
        imgs, anss = mnist.test.next_batch(batchSz)
        sumAcc+=sess.run(accuracy, feed_dict={img: imgs, ans: anss})
    print("  Test Accuracy: %r" % (sumAcc / numBatches))

    sess.close()


if __name__ == "__main__":
    numBatches = 1000
    if len(sys.argv) > 1:
        numBatches = int(sys.argv[1])
    print(">>> Running models using ",str(numBatches)," batches.")
    # fig2_2( numBatches )
    fig2_9( numBatches )
    
    # view_digit( 0, mnist.train.images, mnist.train.labels )
    # view_array( mnist.train.images[:4,:], mnist.train.labels[:4], 2 )

