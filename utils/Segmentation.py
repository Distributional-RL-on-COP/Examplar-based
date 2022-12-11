from scipy import sparse
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import queue
import paint

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
    m = m[0]+m[1]+m[2]  # foreground
    n = n[0]+n[1]+n[2]  # background
    # m = m / m + n
    # n = n / m + n
    m[np.bitwise_and(img[:,:,2] == 255, img[:,:,0] == 0, img[:,:,1] == 0)] = 100
    n[np.bitwise_and(img[:,:,2] == 0, img[:,:,0] == 255, img[:,:,1] == 0)] = 100
    return m.flatten(), n.flatten()


def img2graph(img, acceptable_difference):
    '''
    input: image, np.ndarray
    return: np.lil_matrix
    '''
    h,w = img.shape[:-1]
    a = sparse.lil_matrix((h*w+2, h*w+2))   # h*w, h*w

    # diagonal 1
    diff1 = np.diff(img, axis=1)   
    # print(diff1)  #
    # print(diff1.shape)   # h, w-1, 3
    l1 = np.linalg.norm(diff1, axis=2)  
    # acceptable_difference = 90
    sigma_sq = acceptable_difference / (2*np.log(10))
    l1 = np.exp(-l1 / (2*sigma_sq))    # k = 10000   3,2
    # print(l1)
    # print(l1.shape)   # h, w-1
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
    l2 = np.exp(-l2 / (2*sigma_sq)) 
    l2 = np.append(0, l2)
    l2 = np.append(l2, 0)  
    a.setdiag(l2, k=w)
    a.setdiag(l2, k=-w)
    # m = sparse.csr_matrix(a)
    # m = a.toarray()
    # print(m)    # m is a sparse matrix representing the edges of the graph
    return a

class Graph:
    def __init__(self, graph):
        '''
        graph is of type lil_matrix
        '''
        self.graph = graph
        # self.graph_ = graph.copy()
        self.n = graph.shape[0]

    def BFS(self, start, end, parent): 
        '''judge whether sink is reachable and update self.path by a path from source to sink ''' 
        q = queue.Queue()
        visited = [0] * self.n
        q.put(start)
        visited[start] = 1
        # parent = [-1] * self.n
        while not q.empty():
            x = q.get()
            # print(x)
            rows = self.graph.rows
            data = self.graph.data
            for ind, val in zip(rows[x], data[x]): # ind is vertex and val is weight of edge
                if visited[ind] == 0 and val > 0:
                    q.put(ind)
                    visited[ind] = 1
                    parent[ind] = x
        # print(parent)
        if visited[end] == 1:
            return True
        return False

    # def DFS(self, graph, s, visited):
    #     visited[s] = True
	# 	for i in range(len(graph)):
	# 		if graph[s][i]>0 and not visited[i]:
	# 			self.dfs(graph,i,visited)
    
    def minCut_Fold_Fulkerson(self, source, sink):
        # mask = np.zeros((self.n, self.n))
        parent = [-1] * self.n
        while self.BFS(source, sink, parent):
            node = sink
            minimum = float("Inf")
            data = self.graph.data
            rows = self.graph.rows
            while node != source:
                prev = parent[node]
                value = data[prev][rows[prev].index(node)]
                minimum = min(minimum, value)
                node = prev
            node = sink
            while node != source:
                prev = parent[node]
                data[prev][rows[prev].index(node)] -= minimum
                node = prev

class Segmentater:

    def __init__(self, input_img):
        self.img = input_img
        self.img_ = img.copy()

        self.do_segmentation()
    
    def do_segmentation(self):
        print('---adding scribbles---')
        painter = paint.Painter(self.img, 2)
        painter.paint_mask()

        print('---creating sparse matrix---')
        m, n = get_distribution(self.img, self.img_)
        h,w = self.img.shape[:-1]
        graph = img2graph(self.img_, 90)
        graph[0,1:-1] = m + 1e-250
        graph[1:-1,0] = m + 1e-250
        graph[-1,1:-1] = n + 1e-250
        graph[1:-1,-1] = n + 1e-250
        # print(graph[0])
        print('---graph cutting---')
        g = Graph(graph)
        g.minCut_Fold_Fulkerson(0,h*w+1)
        # print(g.graph[0])
        # print(len(g.graph.data[0]))
        print('---creating mask---')
        mask = np.array(g.graph.data[0]).reshape((h,w))
        mask = np.uint8(mask>0)
        plt.imshow(mask, 'gray')
        plt.show()

if __name__ == "__main__":

    # 10x12
    # img = cv2.imread(r'D:/Courses_2022_Fall/ECE4513/Projects/src/MyCode/utils/segmentation-based-on-graph-cut-and-density-estimation/input1.jpg')
    img = cv2.imread(r"D:\Courses_2022_Fall\ECE4513\Projects\src\MyCode\img\chair\image.jpg")
    # img = img[320:400, 510:580]
    
    seg = Segmentater(img)

