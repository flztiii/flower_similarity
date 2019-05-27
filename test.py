from ImageSimilarity import *
import os
import cv2

if __name__ == "__main__":
    judgement = FlowerSimilarityJudgement()
    judgement.testPrepare("./picture/peony/2.jpeg")
    dataset = os.listdir("./picture/lotus/")
    for image in dataset:
        image = "./picture/lotus/" + image
        score = judgement.test(image)
        print(image, score)
        