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
                    cv2.circle(self.img, (x,y), self.size, colors['red'], -1)  # red: foreground
                else:
                    cv2.circle(self.img, (x,y), self.size, colors['blue'], -1)   # blue: background
                    

    def paint_mask(self):
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.paint_handler)
        while True:
            cv2.imshow('Image', self.img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                self.foreground = not self.foreground
            elif key == ord('q'):   # press q to quit
                break
    