import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ginput
import copy

def get_square_mask(img):
    plt.imshow(img)
    mask_black = np.ones(img.shape)
    x1,y1 = ginput(1)[0]
    x1,y1 = int(x1), int(y1)
    weight = 200
    height = 160
    mask_img = copy.deepcopy(img)
    zeros = np.zeros((height, weight, 3))
    # print(musk_img[y1 -height//2:y1+height//2, x1-weight//2:x1+weight//2])
    mask_img[y1 -height//2:y1+height//2, x1-weight//2:x1+weight//2] = zeros
    mask_black[y1 -height//2:y1+height//2, x1-weight//2:x1+weight//2] = zeros
    # print(musk_img[y1 -height//2:y1+height//2, x1-weight//2:x1+weight//2])
    plt.cla()
    plt.imshow(mask_img)
    plt.savefig("mask_color")
    plt.cla()
    plt.imshow(mask_black)
    plt.savefig("mask_black")
    # plt.show()
    return mask_img, mask_black

if __name__ == "__main__":
    
    new_im = plt.imread("./img\input1.jpg")
    gray_image = cv2.cvtColor(new_im, cv2.COLOR_RGB2GRAY)
    mask_color, mask_black = get_square_mask(new_im)
