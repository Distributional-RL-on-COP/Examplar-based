import cv2 
import matplotlib.pyplot as plt
import numpy as np

from utils.Examplar import Inpainter


if __name__ == "__main__":

    img_src = r"D:\Courses_2022_Fall\ECE4513\Projects\src\MyCode\utils\poission_blending_input\2\coler_mask.jpg"
    mask_src = r"D:\Courses_2022_Fall\ECE4513\Projects\src\MyCode\utils\poission_blending_input\2\black_mask.jpg"

    patch_size = 9

    img = cv2.imread(img_src)
    mask = cv2.imread(mask_src, 0)

    inpainter = Inpainter(img, mask, patch_size, show=True)
    inpainter.exe_inpaint("approx1000.jpg", approx=True, step = 1000)
