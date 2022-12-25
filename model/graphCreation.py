from scipy import sparse
from scipy.stats import norm
import numpy as np
import cv2

def get_distribution(img, img_):
    '''
    img is an image after scribbling
    img_ is the original image
    return m, n
    m is (h x w ,1), probability of image to foreground distribution
    '''
    # foreground
    m = []
    for i in range(3):
        img_foreground = img_[:,:,i][np.bitwise_and(img[:,:,2] == 255, img[:,:,0] == 0, img[:,:,1] == 0)]
        u = np.mean(img_foreground)
        sigma = (np.var(img_foreground))**0.5
        m.append(norm.pdf(img_[:,:,i], u, sigma))

    # background
    n = []
    for i in range(3):
        img_background = img_[:,:,i][np.bitwise_and(img[:,:,2] == 0,img[:,:,0] == 255, img[:,:,1] == 0)]
        u = np.mean(img_background)
        sigma = (np.var(img_background))**0.5
        n.append(norm.pdf(img_[:,:,i], u, sigma))
    m = (m[0]+m[1]+m[2]) 
    n = (n[0]+n[1]+n[2]) 
    # m = m / m + n
    # n = n / m + n
    m[np.bitwise_and(img[:,:,2] == 255, img[:,:,0] == 0, img[:,:,1] == 0)] = 100
    n[np.bitwise_and(img[:,:,2] == 0, img[:,:,0] == 255, img[:,:,1] == 0)] = 100
    return m.flatten(), n.flatten()



def img2graph(img, painted_img, sigma_sq):
    '''
    input: image, np.ndarray
    return: scipy.spase.lil_matrix
    '''
    img = img.astype(np.int32)
    h,w = img.shape[:-1]
    # print(h,w)
    a = sparse.lil_matrix((h*w+2, h*w+2))   # h*w, h*w

    # diagonal 1
    diff1 = np.diff(img, axis=1)  
    # print(diff1)  
    # # print(diff1.shape)   # h, w-1, 3
    l1 = np.linalg.norm(diff1, axis=2)  
    # print(l1**2)
    # # acceptable_difference = 90
    # sigma_sq = 1800
    l1 = 100 * np.exp(-l1**2 / (2*sigma_sq))    # k = 10000   3,2
    l1 = l1.astype(np.int32)
    # print(l1)
    # # print(l1.shape)   # h, w-1
    zeros = np.zeros((h,1), dtype='uint8')
    l1 = np.hstack((l1, zeros)).flatten()
    l1 = np.append(0, l1)
    # print(l1)
    a.setdiag(l1, k=1)
    a.setdiag(l1, k=-1)

    # diagonal 2
    diff2 = np.diff(img, axis=0)
    # print(diff2)
    # print(diff2.shape)   # h-1,w,3
    diff2 = diff2.reshape((h-1)*w, 3)
    # print(diff2)
    l2 = np.linalg.norm(diff2, axis=1)
    # print(l2)
    l2 = 100 * np.exp(-l2**2 / (2*sigma_sq)) 
    l2 = np.append(0, l2)
    l2 = np.append(l2, 0)  
    l2 = l2.astype(np.int32)
    # print(l2)
    a.setdiag(l2, k=w)
    a.setdiag(l2, k=-w)
    # m = sparse.csr_matrix(a)
    # print(m)    # m is a sparse matrix representing the edges of the graph

    background = (np.bitwise_and(painted_img[:,:,2] == 255, painted_img[:,:,0] == 0, painted_img[:,:,1] == 0)).flatten() 
    background = np.insert(background, 0, False)
    # print(background)
    a[-1,background] = 1000
    a[background,-1] = 1000
    foreground = (np.bitwise_and(painted_img[:,:,2] == 0,painted_img[:,:,0] == 255, painted_img[:,:,1] == 0)).flatten()
    foreground = np.insert(foreground, 0, False)
    # print(foreground)
    a[0,foreground] = 1000
    a[foreground,0] = 1000
    return a

if __name__ == "__main__":

    img = np.array([[[0,0,0],[30,30,30],[170,170,170]],
    [[50,50,50],[255,255,255],[200,200,200]]])
    # 
    # img = cv2.imread('/Users/zhaosonglin/Desktop/test1.png')
    # # print(img.shape)
    graph = img2graph(img, 90)
    print(graph.shape)