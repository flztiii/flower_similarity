import os
import cv2
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from joint_bayesian import *
from common import *

def imgresize(img,x,y):
    height = img.shape[0]
    width = img.shape[1]
    l = max(height, width)
    out = np.zeros((l,l,3))
    out[int((l-height)/2):int((l-height)/2)+height, int((l-width)/2):int((l-width)/2)+width] = img
    out = cv2.resize(out,(x,y),cv2.INTER_NEAREST)
    return out

def image_deal(img):
    img = imgresize(img,227,227)
    mean = np.ones_like(img)*128.0
    img = img - mean
    return img

batch_size = 1
weightpath = "./newlog512/checkpoints/model_epoch91.ckpt"
testimg = cv2.imread('z.jpg')
fimg = image_deal(testimg)


# Network params
dropout_rate = 1
num_classes = 18
train_layers = []


x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, num_classes, train_layers)
saver = tf.train.Saver()

score = model.fc8
feature = model.feature

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, weightpath)
    outscore, outfeature = sess.run([score, feature], feed_dict={x: [fimg], keep_prob: dropout_rate})
    outscore = outscore[0]
    outfeature = outfeature[0]
    comfeatures = []
    print outscore
    print outfeature
    for i in range(0,18):
        path = "../"+str(i)+"/0.jpg"
        comi = cv2.imread(path)
        comi = image_deal(comi)
        outs, outf = sess.run([score, feature], feed_dict={x: [comi], keep_prob: dropout_rate})
        comfeatures.append(outf[0])
    result = []
    with open("./new_weights/region0/out/A_con.pkl", "rb") as f:
        A = pickle.load(f)
    with open("./new_weights/region0/out/G_con.pkl", "rb") as f:
        G = pickle.load(f)
    for j in comfeatures:
        j = np.array(j)
        outfeature = np.array(outfeature)
        re = Verify(A,G,outfeature,j)
        result.append(re)
    print(result)

        
