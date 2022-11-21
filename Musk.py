import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ginput
import copy

def get_square_mask(img, width = 200, height = 160):
    plt.imshow(img)

    mask_black = 255*np.zeros(img.shape)
    x1,y1 = ginput(1)[0]
    x1,y1 = int(x1), int(y1)
    plt.show()
    mask_img = copy.deepcopy(img)
    # zeros = np.zeros((height, width, 3))
    ones = np.ones((height, width, 3))

    mask_img[y1 -height//2:y1+height//2, x1-width//2:x1+width//2] = ones
    mask_black[y1 -height//2:y1+height//2, x1-width//2:x1+width//2] = ones
    plt.subplot(131), plt.imshow(img)
    plt.subplot(132), plt.imshow(mask_img)
    plt.subplot(133), plt.imshow(mask_black)
    plt.savefig("masks.jpg")
    plt.show()
    return mask_img, mask_black[:,:,0]
        

if __name__ == "__main__":
    
    new_im = cv2.imread("./img\input1.jpg")
    print(new_im.shape)
    # gray_image = cv2.cvtColor(new_im, cv2.COLOR_RGB2GRAY)
    mask_color, mask_black = get_square_mask(new_im)
    cv2.imwrite("mask_black.jpg", mask_black)
    cv2.imwrite("mask_color.jpg", mask_color)
