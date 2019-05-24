import os
import cv2
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from joint_bayesian import *

def image_deal(img):
    img = cv2.resize(img,(227,227))
    mean = np.ones_like(img)*128.0
    img = img - mean
    return img

batch_size = 1
weightpath = "./new_weights/region7/checkpoints/model_epoch51.ckpt"

# Network params
dropout_rate = 1
num_classes = 12
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
    trainingset = []
    label = []
    for sort in range(0,num_classes):
        for idx in range(0,2000):
            path = "../region7/"+str(sort)+"/"+str(idx)+".jpg"
            testimg = cv2.imread(path)
            fimg = image_deal(testimg)
            outscore, outfeature = sess.run([score, feature], feed_dict={x: [fimg], keep_prob: dropout_rate})
            outfeature = outfeature[0]
            #print(outfeature)
            trainingset.append(outfeature)
            label.append(sort-1)
    trainingset = np.array(trainingset)
    label = np.array(label)
    np.save("./new_weights/region7/out/trainset.npy", trainingset)
    np.save("./new_weights/region7/out/label.npy", label)
    print("saved")
    JointBayesian_Train(trainingset, label,"./new_weights/region7/out/")
    print("done")