import numpy as np

img = np.random.random((10, 10))
zeros = np.zeros((10, 10))
zeros[3:7, 3:7] = np.ones((4, 4))
masked = np.ma.masked_array(img, zeros)
g_x, g_y = np.gradient(img)

print(g_x)
g_y = np.ma.filled(g_y, 0)
g_x = np.ma.filled(g_x, 0)

print(g_x)
print(g_y)