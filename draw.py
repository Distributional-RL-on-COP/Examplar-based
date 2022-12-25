import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    plt.subplot(151), plt.imshow(cv2.cvtColor(cv2.imread(r"img\swim\original.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off'), plt.title("origin")
    plt.subplot(152), plt.imshow(cv2.cvtColor(cv2.imread(r"img\swim\swimimgWithHole.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off'), plt.title("mask")
    plt.subplot(153), plt.imshow(cv2.cvtColor(cv2.imread(r"img\swim\exemplar_done.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off'), plt.title("Exemplar")
    plt.subplot(154), plt.imshow(cv2.cvtColor(cv2.imread(r"img\swim\done.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off'), plt.title("Exemplar + Scene Completion")
    plt.show()