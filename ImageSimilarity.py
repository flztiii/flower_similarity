import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import cv2
import os
import shutil
import re
import pca
from joint_bayesian import *
from common import *

WEIGHT = "./weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
PCA_TRANS = "./weights/pca_trans.npy"
A_CON = "./weights/A_con.pkl"
G_CON = "./weights/G_con.pkl"
TRAIN_PIC_PATH = "./picture/"

BETA = np.array([[1,1,1,1,1,1,1],[1,2,2,2,2,2,1],[1,2,3,3,3,2,1],[1,2,3,4,3,2,1],[1,2,3,3,3,2,1],[1,2,2,2,2,2,1],[1,1,1,1,1,1,1]])
BETA_VALUE = np.sum(BETA)

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
        return self.model.predict(x)[0]

    def calcDistance(self, feature_1, feature_2):
        result = 0.0
        width, height, n_dim = feature_1.shape
        for i in range(0, width):
            for j in range(0, height):
                result = result + float(BETA[i][j])/float(BETA_VALUE) * np.dot(feature_1[i][j], feature_2[i][j].T)/(np.linalg.norm(feature_1[i][j]) * np.linalg.norm(feature_2[i][j]))
        return result

    # 训练
    def train(self):
        # # 加载训练图片特征
        # classes_dir = os.listdir(TRAIN_PIC_PATH)
        # features = []
        # labels = []
        # count = 0
        # for i in range(0, len(classes_dir)):
        #     path_dir = TRAIN_PIC_PATH + classes_dir[i] + "/"
        #     images_path = os.listdir(path_dir)
        #     for image_path in images_path:
        #         image_path = path_dir + image_path
        #         features.append(self.featureExtraction(image_path))
        #         labels.append(i)
        #         shutil.copy(image_path, "./dataset/" + str(count) + image_path[image_path.find(".", 1):])
        #         count = count + 1
        # # PCA降维
        # features = np.array(features)
        # labels = np.array(labels)
        # # 训练联合贝叶斯模型
        # self.trainBayesian(features, labels)
        return

    # 测试前的准备
    def testPrepare(self, raw_image):
        # self.loadBayesianModel()
        self.raw_image_feature = self.featureExtraction(raw_image)
        # self.raw_image_feature = np.dot(self.raw_image_feature, self.trans_matrix)
        return

    # 测试
    def test(self, image_url):
        feature = self.featureExtraction(image_url)
        # feature = np.dot(feature, self.trans_matrix)
        # result = Verify(self.A, self.G, self.raw_image_feature, feature)
        # result = np.dot(self.raw_image_feature, feature.T)/(np.linalg.norm(self.raw_image_feature) * np.linalg.norm(feature))
        result = self.calcDistance(self.raw_image_feature, feature)
        return result

    # # 由数据集提取PCA变换矩阵
    # def extractPCATransMatrix(self, data, n_dim):
    #     low_mat, self.trans_matrix = pca.PCA(data, n_dim)
    #     np.save(PCA_TRANS, self.trans_matrix)
    #     return low_mat
    
    # # 加载PCA变换矩阵
    # def loadPCATransMatrix(self):
    #     self.trans_matrix = np.load(PCA_TRANS)
    #     return
    
    # # 训练联合贝叶斯模型
    # def trainBayesian(self, features, labels):
    #     JointBayesian_Train(features, labels, "./weights/")
    #     return

    # # 加载联合贝叶斯模型
    # def loadBayesianModel(self):
    #     with open(A_CON, "rb") as f:
    #         self.A = pickle.load(f)
    #     with open(G_CON, "rb") as f:
    #         self.G = pickle.load(f)
    #     return

if __name__ == "__main__":
    judgement = FlowerSimilarityJudgement()
    judgement.train()
