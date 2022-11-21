import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

import os

class Inpainter:
    def __init__(self, img:np.ndarray, mask:np.ndarray, patch_size):

        self.img = img
        self.mask = mask
        self.h, self.w = mask.shape[0], mask.shape[1]
        self.patch_size = patch_size
        self.fill_range = copy.deepcopy(mask)
        self.contours = self.find_contours()
        self.confidence = (self.mask==0).astype("float")


    def find_contours(self):
        """
        return the edge contours np.ndarray(point number, 1)
        """
        contours, hierarchy = cv2.findContours(self.fill_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(len(contours[0]))
        contours = contours[0].reshape((-1,2))
        print(contours.shape)

        contours = contours[0].reshape(-1, 2)[:, ::-1]
        # if len(contours) > 1:

        #     for i in range(1, len(contours)):
        #         self.fill_front = np.concatenate((self.fill_front, contours[i].reshape(-1, 2)[:,::-1]), axis=0)

        return contours

    def get_data(self, img, patch:np.ndarray):
        """
        input the patch(2x2) return the image data in the patch
        patch[0, 0]---------patch[1, 0]
        |                   |
        |                   |
        |                   |
        patch[0, 1]---------patch[1, 0]
        """
        return img[patch[0, 0]:patch[0, 1], patch[1, 0]:patch[1, 1]]

    def get_patch(self,point):
        """
        given a point return a patch of size patch_size x patch_size
        """
        k = self.patch_size // 2
        # consider the domain edge
        patch_range = [[max(0, point[0]-k), min(point[0]+k+1, self.h)], 
                       [max(0, point[1]-k), min(point[1]+k+1, self.w)]]
        return np.array(patch_range)

    def update_prioity_C(self):
        for point in self.fill_range:
            patch = self.get_patch(point)
            confidence_sum = self.get_data(self.confidence, patch).sum()
            # Because the domain edge issue, we cannot direct get patch_size x patch_size as a area
            area = (patch[0, 1]-patch[0, 0])*(patch[1, 1]-patch[1, 0])
            self.confidence[point[0], point[1]] = confidence_sum/area

    def get_norm(self):
        """
        get x, y direction derivative, normalize the vector(g_x, g_y)
        """
        g_x = cv2.Scharr(self.fill_range, cv2.CV_64F, 1, 0)
        g_y = cv2.Scharr(self.fill_range, cv2.CV_64F, 0, 1)
        # stack g_x[NxM] and g_y[NxM] -> [N, M, 2]
        normal = np.dstack([g_y, g_x])
        print(normal)
        norm = np.sqrt(g_x * g_x + g_y * g_y).reshape(self.h, self.w, 1).repeat(2, axis=2)
        norm[norm == 0] = 1
        unit_normal = normal/norm
        return unit_normal

    def get_isophote(self):
        pass

    def update_prioity_D(self):
        n_vec = self.get_norm()    
        isophote = self.get_isophote()       

    def update_prioity(self):
        self.update_prioity_C()
        self.update_prioity_D()

    def find_patch(self):
        pass

    def fill_patch(self):
        pass

    def exe_inpaint(self):
        while self.mask.sum() > 0:
            self.find_patch()
            self.fill_patch()

def main(img_src, musk_src, patch_size):

    img = cv2.imread(img_src)
    mask = cv2.imread(musk_src, 0)

    inpainter = Inpainter(img, mask, patch_size)
    inpainter.update_prioity()



if __name__ == "__main__":
    img_src = "D:\Courses_2022_Fall\ECE4513\Projects\src\MyCode\mask_color.jpg"
    mask_src = "D:\Courses_2022_Fall\ECE4513\Projects\src\MyCode\mask_black.jpg"
    main(img_src, mask_src, 9)