from ImageSimilarity import *
import os
import cv2

if __name__ == "__main__":
    judgement = FlowerSimilarityJudgement()
    judgement.testPrepare("./selection/lavender.jpeg")
    dataset = os.listdir("./dataset/")
    for image in dataset:
        image = "./dataset/" + image
        score = judgement.test(image)
        if score > 0.56:
            print(image)
            img = cv2.imread(image)
            cv2.imshow("result", img)
            cv2.waitKey(0)
            