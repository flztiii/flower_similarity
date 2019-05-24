import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

class FlowerSimilarityJudgement:
    def __init__(self):
        self.model = ResNet50(weights='imagenet', include_top=False)
        return
    
    def loadRawImage(self, image_url):
        img = image.load_img(image_url, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        self.raw_features = self.model.predict(x)
        return

    def judge(self, image_url):
        img = image.load_img(image_url, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        compare_features = self.model.predict(x)
        return

if __name__ == "__main__":
