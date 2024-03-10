# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:52:22 2024

@author: Pratham Gupta
"""
#This is for testing of filters and basic image processing

import cv2 
import numpy as np
import matplotlib.pyplot as plt

def find_centroids(dst):
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 
                0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5), 
              (-1,-1),criteria)
    return corners


image = cv2.imread('Q5OG.jpg')
imagetbg = cv2.imread('Q5OG.png')
down_width = 256
down_height = 256
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
kernal= np.array([[1, 0 ,-1], [2, 0, -2],[1, 0,-1]])
kernalT = np.transpose(kernal)
result = cv2.filter2D(resized_down, -1, kernal)
resultT = cv2.filter2D(resized_down,-1, kernalT)
res= cv2.hconcat([resized_down,resultT,result])
cv2.imshow('Res', res)
cv2.waitKey()

gray = cv2.cvtColor(imagetbg, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayed32', gray)
cv2.waitKey()

'''
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Threshold for an optimal value, it may vary depending on the image.
corners= find_centroids(dst)
# To draw the corners

plt.plot(corners)'''