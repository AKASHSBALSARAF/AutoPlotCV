# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:52:22 2024

@author: Pratham Gupta
"""
#This is for testing of filters and basic image processing

import cv2 
import numpy as np
import matplotlib.pyplot as plt


load = cv2.imread('testImages/cropped.png')
loadcopy = cv2.imread('testImages/cropped.png')

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

is_column_zero = np.all(offset == 0, axis=0)
print(is_column_zero)
column_sum=[]
for i in range(down_height):
    column_index = i
    column_sum.append(np.sum(offset[:, column_index]))
    i+=1
print(max(column_sum))
lvline_index =column_sum.index(max(column_sum))+1

is_row_zero = np.all(offset == 0, axis=1)
print(is_row_zero)
row_sum=[]
for i in range(down_width):
    row_index = i
    row_sum.append(np.sum(offset[row_index,:]))
    i+=1
print(max(row_sum))
bhline_index =row_sum.index(max(row_sum))+1

nz=cv2.findNonZero(offset)
a = lvline_index
b = nz[:,0,1].min()
c = nz[:,0,0].max()
d = bhline_index

offset[b,a:c] =120
offset[b:d,a] =120
offset[b:d,c] =120
offset[d,a:c] =120

cv2.imshow('Offset', offset)
cv2.waitKey()
