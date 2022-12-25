import numpy as np
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix
import cv2
from matplotlib import pyplot as plt
import math
import scipy.sparse
from scipy.sparse.linalg import spsolve

import os
import time

# code for different areas in pictire
MASK = 0                 
BOUNDRY = 1
OUTSIDE = 2

def in_mask(location, mask):
    '''
    A function to check if the input location is in the mask region.

    Args:
    location: ndarray (x,y) the location on a image
    mask: ndarray image that represents for mask. 1 for mask region and 0 for unmask region

    Returns
    A bool value 
    True if location in the mask region
    '''
    if mask[location] == 1:
        return True
    elif mask[location] == 0:
        return False
    else:
        return "Please first normalize mask!"

def get_nearby_locations(location):
    '''
    Get the four nearby locations around the input location

    Args:
    location: ndarray with two element

    Return:
    A list contains four tuples 
    '''
    x, y = location
    nearby = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return nearby


def in_boundry(location, mask):
    '''
    A function to detect if the location is on the boundry of mask
    Noted that we define "the boundry of mask" is in the mask
    '''
    if in_mask(location, mask) == False:
        return False
    else:
        for nearby_location in get_nearby_locations(location):
            if in_mask(nearby_location, mask) == False:
                return True
        
        return False
    

def point_location(location, mask):
    '''
    Detect what region s.t. the input location belong to 
    '''
    if in_mask(location,mask) == False:
        return OUTSIDE
    if in_boundry(location,mask) == True:
        return BOUNDRY
    return MASK


def Laplacian(source, location):
    '''
    Apply Laplacian operator to the location on the source image 
    '''
    x, y = location
    value = (4 * source[x,y])- (1 * source[x+1, y]) - (1 * source[x-1, y]) - (1 * source[x, y+1]) - (1 * source[x, y-1])
    return value


def mask_location(mask):
    '''
    Get all location that in the mask region
    '''
    x, y = np.nonzero(mask)
    return list(zip(x, y))


def get_templete_coordinate(mask, coef = 1.2):
    '''
    Function to get the template coordinate according to the mask 
    
    Para
    mask: A normalized mask image

    coef: To determain the size of templete         

    Return
    The four coordinaates 
    '''
    mask_width, mask_height = mask.shape

    x, y = np.nonzero(mask)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2

    # Get d = max - min
    # coef * d to determain the size of templete 
    w = x_max - x_min
    h = y_max - y_min
    w = coef * w
    h = coef * h

    x_min = np.max([x_mid - w/2, 0])
    x_max = np.min([x_mid + w/2, mask_width])
    y_min = np.max([y_mid - h/2, 0])
    y_max = np.min([y_mid + h/2, mask_height])

    return int(x_min), int(x_max), int(y_min), int(y_max)


# Create the A sparse matrix
def poisson_sparse_matrix(points):
    '''
    Create a sparse A matrix 
    '''
    # N = number of points in mask
    N = len(points)
    A = lil_matrix((N,N))
    
    for i,index in enumerate(points):
        A[i,i] = 4

        for x in get_nearby_locations(index):
            if x not in points: 
                continue
            j = points.index(x)
            A[i,j] = -1

    return A



def process(source, target, mask):
    locations = mask_location(mask)
    N = len(locations)
    # Create poisson A matrix. Contains mostly 0's, some 4's and -1's
    A = poisson_sparse_matrix(locations)

    # Create B matrix
    b = np.zeros(N)
    for i,location in enumerate(locations):
        # bi is the div of each point in mask
        b[i] = Laplacian(source, location)

        # if mask point is on the boundry 
        # add the constrain of target image
        if point_location(location, mask) == BOUNDRY:
            for pt in get_nearby_locations(location):
                if in_mask(pt,mask) == False:
                    b[i] += target[pt]

    x = linalg.cg(A, b)
    # Copy target photo, make sure as int
    composite = np.copy(target).astype(int)
    # Place new intensity on target at given index
    for i,location in enumerate(locations):
        composite[location] = x[0][i]
    return composite


def normalize_mask(mask_img):
    # Normalize mask to range [0,1]
    mask = mask_img.astype(np.float64) / 255.
    # Make mask binary
    mask[mask != 1] = 0
    mask[mask != 0] = 1

    # Trim to one channel
    mask = mask[:,:,0]

    return mask


def poisson_blending(source_img, target_img, mask):
    print("Start to poisson blending...")
    num_channels = source_img.shape[-1]

    temp = []
    for i in range(num_channels):
        result = process(source_img[:,:,i], target_img[:,:,i], mask)
        temp.append(result)
        print(f"Finish channel {i}")
    # Merge the channels back into one image
    result = cv2.merge((temp[0], temp[1], temp[2]))

    print("Finish poisson blending")
    # Write result
    return result

def multi_matching(original_img, matching_dir:str, mask, show_matching_part = False):
    
    min_SSD = float("inf")
    cnt = 1
    start = time.time()
    for filename in os.listdir(r"./"+matching_dir):
        print("match picture search number {}".format(cnt))
        cnt += 1
        matching_img = cv2.imread(matching_dir+"/"+filename)
        matching_img = cv2.cvtColor(matching_img, cv2.COLOR_BGR2RGB)

        x_min, x_max, y_min, y_max = get_templete_coordinate(mask, coef=1.5)

        # Get the mask for template
        template_mask = (mask == 0).astype(np.uint8)
        template_mask = template_mask[x_min:x_max, y_min:y_max]

        if(x_max-x_min > matching_img.shape[0] or y_max-y_min > matching_img.shape[1]):
            continue

        # Do the template matching
        res_list = []
        for i in range(3):
            matching_img_ = matching_img[:,:,i]
            template = original_img[x_min:x_max, y_min:y_max, i]
            x = cv2.matchTemplate(matching_img_, template, cv2.TM_SQDIFF, mask = template_mask)
            res_list.append(x)

        res = sum(res_list) / 3
        (h, w) = template.shape
        ssd_i = np.min(res)
        if(min_SSD > ssd_i):
            min_SSD = ssd_i
            loc = np.where(res == np.min(res))
            for pt in zip(*loc[::-1]):
                if show_matching_part == True:
                    cv2.rectangle(matching_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
                    plt.imshow(matching_img)

            best_matching = matching_img[pt[1]:pt[1] + h,pt[0]:pt[0] + w]
            template_in_original = original_img[x_min:x_max, y_min:y_max]
    end = time.time()
    print("Finish the Multi-matching use {} seconds".format(end-start))
    return best_matching, template_in_original, template_mask    

def template_matching(original_img, matching_img, mask, show_matching_part = False):
    '''
    Function to perform template matching

    Args:

    original_img: The RGB original image with mask 

    matching_img: The RGB matching image

    mask: 2_D Normal mask for orginal image. The unwanted region is labelled as 1

    show_matching_part: True will show the best matching patch (rectangle) in the matching image 
    
    Return

    best_matching: The area in the matching_img such that best matches the template

    template_in_original: template region in the original image

    template_mask: Because the orginal image has mask. we do not want the mask region
                   in the template affect the similar score 

    '''

    x_min, x_max, y_min, y_max = get_templete_coordinate(mask, coef=1.5)

    # Get the mask for template
    template_mask = (mask == 0).astype(np.uint8)
    template_mask = template_mask[x_min:x_max, y_min:y_max]

    # Do the template matching
    res_list = []
    for i in range(3):
        matching_img_ = matching_img[:,:,i]
        template = original_img[x_min:x_max, y_min:y_max, i]
        x = cv2.matchTemplate(matching_img_, template, cv2.TM_SQDIFF, mask = template_mask)
        res_list.append(x)
    
    res = sum(res_list) / 3

    (h, w) = template.shape

    # show the similarity
    print(f"The SSD score is {np.min(res)}")

    # if want to see the matching patch in the matching image 
    # show the rectangle patch in the matching image 
    loc = np.where(res == np.min(res))
    for pt in zip(*loc[::-1]):
        if show_matching_part == True:
            cv2.rectangle(matching_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
            plt.imshow(matching_img)

    best_matching = matching_img[pt[1]:pt[1] + h,pt[0]:pt[0] + w]
    template_in_original = original_img[x_min:x_max, y_min:y_max]

    print("Finish the templete matching")
    return best_matching, template_in_original, template_mask

def three_dimention_mask(mask):
    (x,y) = mask.shape
    three_mask = np.zeros((x,y,3))
    for i in range(3):
        three_mask[:,:,i] = mask
    return three_mask

def enlarge_mask(mask):
    larger_mask = np.zeros_like(mask)
    x, y = np.nonzero(mask)
    x_min = math.floor(np.min(x)) - 1
    x_max = math.ceil(np.max(x)) + 2
    y_min = math.floor(np.min(y)) - 1 
    y_max = math.ceil(np.max(y)) + 2
    larger_mask[x_min:x_max, y_min:y_max] = 1
    # print(x_min, x_max, y_min, y_max)


    return larger_mask

def impaint(original_img, mask, blending_img):
    x_min, x_max, y_min, y_max = get_templete_coordinate(mask, coef=1.5)
    original_img[x_min:x_max, y_min:y_max] = blending_img
    return original_img

def laplacian_matrix(n, m):   
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m) # [i, i+m]
    mat_A.setdiag(-1, -1*m)  # [i, i-m]
    
    return mat_A

def construct_sparse_A(source, target, mask):
    y_max, x_max = target.shape[:-1]
    Matrix_A = laplacian_matrix(y_max, x_max)
    laplacian = Matrix_A.tocsc()
    for y in range(1, y_max - 1):
        for x in range(1, x_max - 1):
            if mask[y, x] == 0:
                k = x + y * x_max
                Matrix_A[k, k] = 1
                Matrix_A[k, k + 1] = 0
                Matrix_A[k, k - 1] = 0
                Matrix_A[k, k + x_max] = 0
                Matrix_A[k, k - x_max] = 0
    Matrix_A = Matrix_A.tocsc()

    return Matrix_A, laplacian


def poisson_blending_fast(source, target, mask):
    start = time.time()
    y_max, x_max, z_max = target.shape[:]
    A, laplacian = construct_sparse_A(source, target, mask)

    mask_flat = mask.flatten()    
    for channel in range(z_max):
        source_flat = source[:, :, channel].flatten()
        target_flat = target[:, :, channel].flatten()        
    
        alpha = 1
        b = laplacian.dot(source_flat) * alpha
        b[mask_flat == 0] = target_flat[mask_flat == 0]

        x = spsolve(A, b)    
        x = x.reshape((y_max, x_max))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        
        target[:, :, channel] = x
    end = time.time()
    print("It takes ", end-start," seconds to fininsh Poisson blending size:", mask.shape)
    return target

if __name__ == "__main__":

    # 这些理论上都不需要，前面你已经读取过了
    mask = cv2.imread("scene completion image/2/forest_10000_mask.jpg")
    mask = normalize_mask(mask)

    matching_img = cv2.imread("scene completion image/2/matching.jpg")
    matching_img = cv2.cvtColor(matching_img, cv2.COLOR_BGR2RGB)

    original_img = cv2.imread("scene completion image/2/forest_10000.jpg")
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # 可以直接在这里跑template matching, 可以看一下注释里面对输入的要求
    best_matching, template_in_original, template_mask = template_matching(original_img, matching_img, mask)


    # 对上面的图片进行可视化
    mask_ = template_mask == 0
    l_mask = enlarge_mask(mask_)

    plt.subplot(2,2,1)
    plt.imshow(best_matching)
    plt.subplot(2,2,2)
    plt.imshow(template_in_original)
    plt.subplot(2,2,3)
    plt.imshow(mask_,"gray")
    plt.subplot(2,2,4)
    plt.imshow(l_mask,"gray")
    plt.show()

    # 因为原图片在mask部位是黑色的，但是我们的mask
    # 因为一些computational error不是正正好好每一个都和原图片上的mask对应
    r = poisson_blending_fast(best_matching, template_in_original, l_mask)
    plt.imshow(r)
    plt.show()

    impaint_img = impaint(original_img, mask, r)
    plt.imshow(impaint_img)
    plt.show()
