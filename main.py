import cv2 
import matplotlib.pyplot as plt
import numpy as np


class ExpInpaint:
    def __init__(self, img:np.ndarray, musk:np.ndarray):
        self.img = img
        self.grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        self.musk = musk
        self.injury = self.get_musked(self.grey, musk)
        # plt.imshow(self.injury)
        pass

    def get_musked(self, img, musk):
        if(img.shape != musk.shape):
            return
        return img*musk
        
if __name__ == "__main__":
    img = cv2.imread("MyCode\img\input1.jpg")
    # cv2.imshow("img",img)
    musk = cv2.imread("MyCode\img\input1_mask.jpg")
    examplarBasedInpaint = ExpInpaint(img, musk)
    plt.show()