# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:52:22 2024

@author: Pratham Gupta
"""
#This is for testing of filters and basic image processing

import cv2 
import numpy as np
import matplotlib.pyplot as plt


load = cv2.imread('testImages/Q5OG.jpg')
loadcopy = cv2.imread('testImages/Q5OG.jpg')

down_width = 720
down_height = 720
down_points = (down_width, down_height)
image = cv2.resize(load, down_points, interpolation= cv2.INTER_LINEAR)
imagecopy = cv2.resize(loadcopy, down_points, interpolation= cv2.INTER_LINEAR)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
 
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
 
# Threshold for an optimal value, it may vary depending on the image.
image[dst>0.012*dst.max()]=[255,255,255]

#Till here, the code marks all the corners with a white 4 point box.

offset = cv2.cvtColor(image-imagecopy, cv2.COLOR_BGR2GRAY)


#Here, the code finds the difference between the image and the corner added image, so it gives the corners only
nz=cv2.findNonZero(offset)
nzn = nz[0:18]
b = np.reshape(nzn,(6,6))
print(nzn)

a = nz[:,0,0].min()
b = nz[:,0,1].min()
c = nz[:,0,0].max()
d = nz[:,0,1].max()

offset[b,a:c] =120
offset[b:d,a] =120
offset[b:d,c] =120
offset[d,a:c] =120

cv2.imshow('Offset', offset)
cv2.waitKey()
