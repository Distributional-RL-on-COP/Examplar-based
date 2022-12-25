from scipy import sparse
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import graphCreation
import graphCutSparse
import cv2
import paint

# k = 10
# img = np.array([[[0,0,0],[30,30,30],[170,170,170]],
# [[50,50,50],[255,255,255],[200,200,200]],
# [[20,20,20],[200,200,200],[140,140,140]]])

def segment(path, scale = 0.5, sigma_sq = 2700):
    img = cv2.imread(path)
    print(img.shape)
    H,W = img.shape[:2]
    width = int(W * scale)
    height = int(H * scale)
    print(width)
    print(height)
    dimensions = (width, height)
    print(dimensions)
    img_ = cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
    print(img_.shape)
    # Select ROI, Crop image
    r = cv2.selectROI(img_)
    imgCrop = img_[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    print(imgCrop.shape)
    # imgCrop = img[340:380, 530:580]
    imgCrop_ = imgCrop.copy()
    
    print('---adding scribbles---')
    painter = paint.Painter(imgCrop, 2)
    painter.paint_mask()
    cv2.imwrite('painted.jpg', imgCrop)

    print('---creating sparse matrix---')
    h,w = imgCrop.shape[:-1]
    # print(h, w)
    graph = graphCreation.img2graph(imgCrop_, imgCrop, sigma_sq)
    # print(graph[0])
    print('---graph cutting---')
    g = graphCutSparse.Graph(graph)
    g.minCut_Fold_Fulkerson(0,h*w+1)
    # print(g.graph)
    # print(len(g.graph.data[0]))
    print('---creating mask---')
    maskCrop = g.get_mask()[1:-1]
    maskCrop = np.array(maskCrop).reshape(h,w)
    mask = np.zeros_like(img_[:,:,0])
    mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = -maskCrop
    mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
    composite1 = mask/255 * img[:,:,0]
    composite2 = mask/255 * img[:,:,1]
    composite3 = mask/255 * img[:,:,2]
    composite = np.zeros_like(img)
    composite[:,:,0] = composite1
    composite[:,:,1] = composite2
    composite[:,:,2] = composite3
    cv2.imwrite('composite.jpg', composite)
    return mask

if __name__ == "__main__":
    path = '/Users/zhaosonglin/Desktop/test/test3/original3.jpg'
    mask = segment(path, 0.3, 1800)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.dilate(mask ,kernel,iterations=1)
    # kernel = cv2.
    # mask = 
    # mask = 255 * mask
    # print(mask[455:470,455:465])
    cv2.imwrite('mask.jpg', mask)
    plt.imshow(mask, 'gray')
    plt.show()