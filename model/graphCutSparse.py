import queue
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import sparse
import graphCreation

class Graph:
    def __init__(self, graph):
        '''
        graph is of type lil_matrix
        '''
        self.graph = graph
        self.graph_ = graph.copy()
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

    def get_mask(self):
        q = queue.Queue()
        visited = [0] * self.n
        q.put(0)  # start from source 
        visited[0] = 1
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
        return visited


if __name__ == "__main__":

    img = np.array([[[0,0,0],[30,30,30],[170,170,170]],
    [[50,50,50],[255,255,255],[200,200,200]],
    [[20,20,20],[200,200,200],[140,140,140]]])
    graph = graphCreation.img2graph(img, img, 2700)
    print(graph)
    g = Graph(graph)
    print(g,graph)
    g.minCut_Fold_Fulkerson(0,8)
    mask = g.get_mask()
    print(mask)
    plt.imshow(mask)
    plt.show()
    print(g.graph)
