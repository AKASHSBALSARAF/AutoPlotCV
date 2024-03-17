# Importing all required packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
 
# Read in the cascade classifiers for face and eyes
graph_cascade = cv2.CascadeClassifier('plotCasscade/classifier/cascade.xml')
 
# create a function to detect face
def adjusted_detect_graph(img):
     
    graph_img = img.copy()
    graph_rect = graph_cascade.detectMultiScale(graph_img, scaleFactor = 1.2, minNeighbors = 3)
     
    for (x, y, w, h) in graph_rect:
        cv2.rectangle(graph_img, (x, y), (x + w, y + h),(0,255,0),5)
    return graph_img
 
 
# Reading in the image and creating copies
img = cv2.imread('output/graph4.png')
img_copy1 = img.copy()
 
# Detecting the face 
face = adjusted_detect_graph(img_copy1)
plt.imshow(face)
# Saving the image
cv2.imshow('graph.png', face)
cv2.waitKey(0)