from ImageSimilarity import *
import os
import cv2

if __name__ == "__main__":
    judgement = FlowerSimilarityJudgement()
    judgement.testPrepare("./picture/chrysanthemum/download.jpeg")
    dataset = os.listdir("./picture/chrysanthemum/")
    for image in dataset:
        image = "./picture/chrysanthemum/" + image
        score = judgement.test(image)
        if score > 0.58
        