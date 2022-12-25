import numpy as np
import cv2
import matplotlib.pyplot as plt
import segmentatioin 
from Examplar import Inpainter 
from scene_completion import *
from Musk import click_square_mask

import argparse

def three_dimention_mask(mask):
    mask = 1-mask
    (x,y) = mask.shape
    three_mask = np.zeros((x,y,3))
    for i in range(3):
        three_mask[:,:,i] = mask
    return three_mask

def get_mask(img:np.ndarray, mask:np.ndarray):

    r, g, b = cv2.split(img)
    r = r*(1-mask)
    g = g*(1-mask)
    b = b*(1-mask)

    return cv2.merge((b,g,r))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='original picture')
    parser.add_argument('--mask_type', type=str, help='cut or mask')
    parser.add_argument('--ratio', type=float, help='input the inpaint ratio of two\
                                    strategies')
    parser.add_argument('--img_path', type=str, help="input the path of the original \
                                    picture")
    parser.add_argument('--mask_path', type=str, help="input the path of the mask")
    parser.add_argument('--match_path', type=str, help="input the path of the matching \
                                    pictures")
    parser.add_argument("--write_path", type=str, help="select a path to store pictures")
    args = parser.parse_args()

    if(args.img_path == None):
        img_src = r"img\sling\original.jpg"
    else:
        img_src = args.img_path
    img = cv2.imread(img_src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img), plt.title("original")
    plt.show()

    # cut mode
    if(args.mask_type == "cut"):
        mask_1 = segmentatioin.segment(img_src, scale=0.5, sigma_sq=2700)
        mask = np.stack((mask_1, mask_1, mask_1), axis=2)
    # read mask mode
    else:
        if(args.mask_path == None):
            mask_src = r"img\sling\mask.jpg"
        else:
            mask_src = args.mask_path
        mask = cv2.imread(mask_src)
        print(mask.shape)
        print(np.unique(mask))
        plt.imshow(mask), plt.title("mask")
        plt.show()
    # set match path
    if(args.match_path == None):
        matching_dir = r"img\sling\match"
    else:
        matching_dir = args.match_path
    # set write path
    if(args.write_path == None):
        write_path = r"img\sling\\"
    else:
        write_path = args.write_path

    # The mask max is 1 not 255
    mask[mask>0] = 1
    mask_3D = 1-mask
    imgWithHole = img*(mask_3D)

    plt.subplot(121), plt.imshow(mask_3D), plt.title("mask_3D")
    print(np.unique(mask_3D))
    plt.subplot(122), plt.imshow(imgWithHole), plt.title("imgWithHole")
    plt.show()
    
    ratio = 0.1 if args.ratio==None else args.ratio
    
    # Examplar 
    patch_size = 5
    inpainter = Inpainter(np.array(imgWithHole,np.uint8), mask[:, :, 0], patch_size, show=False, comp_rate = ratio)
    fill_image, fill_range = inpainter.exe_inpaint("approx10000", approx=True, step = 1000)
    plt.subplot(121), plt.imshow(fill_image), plt.title("fill_image")
    plt.subplot(122), plt.imshow(fill_range, "gray"), plt.title("fill_range")
    plt.show()
    
    if(ratio < 1):
        # Template matching
        mask = fill_range
        original_img = fill_image

        best_matching, template_in_original, template_mask = multi_matching(original_img, matching_dir, mask)

        # visualization all images above
        mask_ = template_mask == 0
        l_mask = enlarge_mask(mask_)

        plt.subplot(2,2,1)
        plt.imshow(best_matching)
        plt.title("Best matching")
        plt.subplot(2,2,2)
        plt.imshow(template_in_original)
        plt.title("template in original")
        plt.show()
        
        
        r = poisson_blending_fast(best_matching, template_in_original, mask_)
        plt.imshow(r)
        plt.title("Poisson result")
        plt.show()

        impaint_img = impaint(original_img, mask, r)

        plt.imshow(impaint_img)
        plt.show()
        impaint_img = cv2.cvtColor(impaint_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(write_path+"\hybrid_done.jpg", impaint_img)

        # inpainter = Inpainter(fill_image, fill_range, 5, show=False, comp_rate=1)
        # inpainter.exe_inpaint("examplar", approx=True)
    else:
        fill_image = cv2.cvtColor(fill_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(write_path+"\exemplar_done.jpg", fill_image)
