import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import cv2

WEIGHT = "./weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

class FlowerSimilarityJudgement:
    def __init__(self):
        self.model = ResNet50(weights=WEIGHT, include_top=False)
        return
    
    def loadRawImage(self, image_url):
        img = image.load_img(image_url, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        self.raw_features = self.model.predict(x).reshape(((1,100352)))
        return

    def judge(self, image_url):
        img = image.load_img(image_url, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        compare_features = self.model.predict(x).reshape((1,100352))
        return self.cosSimilarity(self.raw_features, compare_features)
    
    def cosSimilarity(self, vector_1, vector_2):
        vec_norm_1 = np.linalg.norm(vector_1)
        vec_norm_2 = np.linalg.norm(vector_2)
        return np.dot(vector_1,vector_2.T)/(vec_norm_1*vec_norm_2)

if __name__ == "__main__":
    judgement = FlowerSimilarityJudgement()
    judgement.loadRawImage("./picture/1.jpg")
    print(judgement.judge("./picture/2.jpeg"))
