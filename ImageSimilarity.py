import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import cv2
import os
import PCA
from joint_bayesian import *
from common import *

WEIGHT = "./weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
PCA_TRANS = "./weights/pca_trans.npy"
A_CON = "/weights/A_con.pkl"
G_CON = "/weights/G_con.pkl"

class FlowerSimilarityJudgement:
    def __init__(self):
        self.model = ResNet50(weights=WEIGHT, include_top=False)
        return
    
    # 加载图像特征
    def featureExtraction(self, image_url):
        img = image.load_img(image_url, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x).reshape((1,100352))

    # 训练
    def train(self):
        # 加载训练图片
        return

    # 测试前的准备
    def testPrepare(self,raw_image):
        self.loadPCATransMatrix()
        self.loadBayesianModel()
        self.raw_image_feature = self.featureExtraction(raw_image)
        self.raw_image_feature = np.dot(self.raw_image_feature, self.trans_matrix)
        return

    # 测试
    def test(self, image_url):
        feature = self.featureExtraction(image_url)
        feature = np.dot(feature, self.trans_matrix)
        result = Verify(self.A, self.G, self.raw_image_feature, feature)
        return result

    # 由数据集提取PCA变换矩阵
    def extractPCATransMatrix(self，data, n_dim):
        _, self.trans_matrix = PCA.PCA(data, ndim)
        np.save(PCA_TRANS, self.trans_matrix)
        return
    
    # 加载PCA变换矩阵
    def loadPCATransMatrix(self):
        self.trans_matrix = np.load(PCA_TRANS)
        return
    
    # 训练联合贝叶斯模型
    def trainBayesian(self):
        JointBayesian_Train(trainingset, label,"./weights/")
        return

    # 加载联合贝叶斯模型
    def loadBayesianModel(self):
        with open(A_CON, "rb") as f:
            self.A = pickle.load(f)
        with open(G_CON, "rb") as f:
            self.G = pickle.load(f)
        return

if __name__ == "__main__":
    judgement = FlowerSimilarityJudgement()
