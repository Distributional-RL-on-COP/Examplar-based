import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

'''
foreground: red (0,255,0)
background: blue (255,0,0)
'''

class Painter:
    def __init__(self, img, size):
        self.img = img
        self.size = size
        self.drawing = False
        self.foreground = True
    
    def paint_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print("event: EVENT_LBUTTONDOWN")
            self.drawing = not self.drawing
            # print(self.drawing)
        if event == cv2.EVENT_MOUSEMOVE:
            # print("event: EVENT_MOUSEMOVE")
            # print('move')
            if self.drawing == True:
                # print('draw')
                if self.foreground:
                    cv2.circle(self.img, (x,y), self.size, colors['blue'], -1)  # red: foreground
                else:
                    cv2.circle(self.img, (x,y), self.size, colors['red'], -1)   # blue: background
                    

    def paint_mask(self):
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.paint_handler)
        while True:
            cv2.imshow('Image', self.img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                self.foreground = not self.foreground
            if key == ord('a'):
                self.size += 1
            if key == ord('b') and self.size >= 2:
                self.size -= 1
            elif key == ord('q'):   # press q to quit
                break
    




# img = cv2.imread('/Users/zhaosonglin/Desktop/test.jpeg')
# img_ = img.copy()
# painter = Painter(img, 3)
# painter.paint_mask()

# m = []
# for i in range(3):
#     img_foreground = img_[:,:,i][np.bitwise_and(img[:,:,2] == 255, img[:,:,0] == 0, img[:,:,1] == 0)]
#     u = np.mean(img_foreground)
#     sigma = (np.var(img_foreground))**0.5
#     m.append(norm.pdf(img_, u, sigma))

# # background
# n = []
# for i in range(3):
#     img_background = img_[:,:,i][np.bitwise_and(img[:,:,2] == 0,img[:,:,0] == 255, img[:,:,1] == 0)]
#     u = np.mean(img_background)
#     sigma = (np.var(img_background))**0.5
#     n.append(norm.pdf(img_, u, sigma))
# m = np.minimum(m[0],m[1],m[2])  # foreground
# n = np.minimum(n[0],n[1],n[2])  # background
# m[np.bitwise_and(img[:,:,2] == 255, img[:,:,0] == 0, img[:,:,1] == 0)] = 100
# n[np.bitwise_and(img[:,:,2] == 0, img[:,:,0] == 255, img[:,:,1] == 0)] = 100
# print(m)
# print(n)






# # foreground
# fore = []
# for i in range(3):
#     img_foreground = img_[:,:,i][np.bitwise_and(img[:,:,2] == 255, img[:,:,0] == 0, img[:,:,1] == 0)]
#     u = np.mean(img_foreground)
#     sigma = (np.var(img_foreground))**0.5
#     fore.append((u, sigma))

# # background
# back = []
# for i in range(3):
#     img_background = img_[:,:,i][np.bitwise_and(img[:,:,2] == 0,img[:,:,0] == 255, img[:,:,1] == 0)]
#     u = np.mean(img_background)
#     sigma = (np.var(img_background))**0.5
#     back.append((u, sigma))


# print(fore)
# print(back)


    


    


# img1_foreground = img_[:,:,0][np.bitwise_and(img[:,:,2] == 255,img[:,:,0] == 0, img[:,:,1] == 0)]
# img2_foreground = img_[:,:,1][np.bitwise_and(img[:,:,2] == 255,img[:,:,0] == 0, img[:,:,1] == 0)]
# img3_foreground = img_[:,:,2][np.bitwise_and(img[:,:,2] == 255,img[:,:,0] == 0, img[:,:,1] == 0)]
# img1_background = img_[:,:,0][np.bitwise_and(img[:,:,2] == 0,img[:,:,0] == 255, img[:,:,1] == 0)]
# img2_background = img_[:,:,1][np.bitwise_and(img[:,:,2] == 0,img[:,:,0] == 255, img[:,:,1] == 0)]
# img3_background = img_[:,:,2][np.bitwise_and(img[:,:,2] == 0,img[:,:,0] == 255, img[:,:,1] == 0)]

# print(img1_foreground.shape)


# plt.hist(img1_foreground.flatten(), bins=256, range=[0,256])
# plt.hist(img1_background.flatten(), bins=256, range=[0,256])
# plt.show()

# test = img_[:,:,0][np.bitse_and(img_[:,:,2] == 0, img[:,:,0] - img[:,:,1] == 255]
# plt.hist(test.flatten(), bins=256, range=[0,256])

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
# plt.imshow(img)
# plt.show()