import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ginput
import copy

def click_square_mask(img, width = 200, height = 160):
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

def get_mask(img:np.ndarray, mask:np.ndarray):
    # threadhold
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    res = np.zeros(mask.shape)

    black_mask = (gray_mask>50)*np.ones(gray_mask.shape)

    r, g, b = cv2.split(img)
    r = r*(1-black_mask)
    g = g*(1-black_mask)
    b = b*(1-black_mask)

    return black_mask, cv2.merge((b,g,r))
        
if __name__ == "__main__":
    
    # new_im = cv2.imread("./img\input1.jpg")
    # print(new_im.shape)
    # # gray_image = cv2.cvtColor(new_im, cv2.COLOR_RGB2GRAY)
    # mask_color, mask_black = click_square_mask(new_im)
    # cv2.imwrite("mask_black.jpg", mask_black)
    # cv2.imwrite("mask_color.jpg", mask_color)

    img = cv2.imread(r"img\bird\bird_origin.jpg")
    # mask_black, mask_color = get_mask(img, mask)
    mask_black, mask_color = click_square_mask(img)
    plt.imshow(mask_color)
    plt.show()
    cv2.imwrite("black_mask.jpg", mask_black)
    cv2.imwrite("coler_mask.jpg", mask_color)
