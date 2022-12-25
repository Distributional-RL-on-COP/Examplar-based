import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ginput
import copy

def click_square_mask(img, width = 200, height = 160):
    N_rows, N_cols, _ = img.shape
    plt.imshow(img)

    mask_black = 255*np.zeros(img.shape)
    x1,y1 = ginput(1)[0]
    x1,y1 = int(x1), int(y1)
    mask_img = copy.deepcopy(img)
    zeros = np.zeros((height, width, 3))
    plt.show()

    ones = np.ones((height, width, 3))
    mask_img[y1 -height//2:y1+height//2, x1-width//2:x1+width//2] = ones
    mask_black[y1 -height//2:y1+height//2, x1-width//2:x1+width//2] = ones

    plt.imshow(mask_img)
    plt.show()
    # plt.subplot(131), plt.imshow(img)
    # plt.subplot(132), plt.imshow(mask_img)
    # plt.subplot(133), plt.imshow(mask_black)
    # plt.savefig("masks.jpg")
    # plt.show()
    return mask_img, mask_black

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
    
    dir_name = r"img\baby\\"
    img = cv2.imread(dir_name+ r"original.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # mask_black, mask_color = get_mask(img, mask)
    mask_color, mask_black = click_square_mask(img, 300, 500)
    plt.imshow(mask_color)
    plt.show()
    cv2.imwrite(dir_name+"black_mask.jpg", mask_black)
    cv2.imwrite(dir_name+"color_mask.jpg", mask_color)
