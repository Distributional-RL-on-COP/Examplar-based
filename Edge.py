import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_adj(img:np.ndarray, edge_queue:list):
    # give a point find the edge adjacent to it
    temp = edge_queue[-1]
    if(len(edge_queue)>=2):
        prev = edge_queue[-2]
    else:
        prev = None
    i, j = temp[0], temp[1]
    adj_ls = []
    if(i-1 >= 0):           # up
        adj_ls.append([i-1, j])
    if(j+1<img.shape[1]):   # right
        adj_ls.append([i, j+1])
    if(i+1<img.shape[0]):   # down
        adj_ls.append([i+1, j])
    if(j-1>=0):             # left
        adj_ls.append([i, j-1])
    for next in adj_ls:
        print(img[next[0], next[1],:])
        if( img[next[0], next[1], 0] <= 10 and 
            img[next[0], next[1], 1] <= 10 and
            img[next[0], next[1], 2] <= 10):
            if(prev == None or (prev[0]!=next[0] or prev[1]!=next[1])):
                return next

def get_edge(img:np.ndarray):
    # search for the first edge pixel

    edge_queue = []

    for i in range(img.shape[0]):
        if(len(edge_queue)!=0):
            break
        for j in range(img.shape[1]):
            if( img[i, j, 0] == 0 and 
                img[i, j, 1] == 0 and
                img[i, j, 2] == 0):
                edge_queue.append([i, j])

                break

    head = edge_queue[0]
    print(head)
    temp = find_adj(img, edge_queue)
    print(temp)
    while((temp[0]==head[0] and temp[1]==head[1]) == False):
        edge_queue.append(temp)
        temp = find_adj(img, edge_queue)
        print(temp)
    return edge_queue

if __name__ == "__main__":
    src = "mask_black.jpg"
    img = cv2.imread(src)
    pic = 256*np.ones((50, 50, 3))
    pic[20:25, 10:15, :] = np.zeros((5, 5, 3))
    edge_queue = get_edge(pic)