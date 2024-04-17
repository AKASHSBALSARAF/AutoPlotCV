# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:52:22 2024

@author: Pratham Gupta
"""
#This is for testing of filters and basic image processing

import cv2 
import numpy as np
import matplotlib.pyplot as plt


load = cv2.imread('testImages/test2.png')
loadcopy = cv2.imread('testImages/test2.png')

down_width = 720
down_height = 720
down_points = (down_width, down_height)
image = cv2.resize(load, down_points, interpolation= cv2.INTER_LINEAR)
imagecopy = cv2.resize(loadcopy, down_points, interpolation= cv2.INTER_LINEAR)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
dst = cv2.Canny(gray,400,500)
 
# Threshold for an optimal value, it may vary depending on the image.
image[dst>0.012*dst.max()]=[255,255,255]

offset = cv2.cvtColor(image-imagecopy, cv2.COLOR_BGR2GRAY)
'''cv2.imshow('Offset',offset)
cv2.waitKey()'''

column_sum=[]
for i in range(down_height):
    column_index = i
    column_sum.append(np.sum(offset[:, column_index]))
    i+=1
print(column_sum)
print(max(column_sum))
lvline_index =column_sum.index(max(column_sum))

loopline2_index=lvline_index
while abs(loopline2_index-lvline_index)<10:
   column_sum.pop(loopline2_index)
   loopline2_index = column_sum.index(max(column_sum))
columntestline_index = loopline2_index

if columntestline_index < lvline_index:
    vline_index = columntestline_index
else:
    vline_index=columntestline_index+1

row_sum=[]
for i in range(down_width):
    row_index = i
    row_sum.append(np.sum(offset[row_index,:]))
    i+=1
bhline_index =row_sum.index(max(row_sum))

loopline_index=bhline_index
while abs(loopline_index-bhline_index)<10:
   row_sum.pop(loopline_index)
   loopline_index = row_sum.index(max(row_sum))
rowtestline_index = loopline_index

if rowtestline_index < bhline_index:
    hline_index = rowtestline_index
else:
    hline_index=rowtestline_index+1


a = lvline_index
b = hline_index  # nz[:,0,1].min()
c = vline_index #nz[:,0,0].max()
d = bhline_index

offset[b,a:c] =128
offset[b:d,a] =128
offset[b:d,c] =128
offset[d,a:c] =128
cv2.imshow('Offset', offset)
cv2.waitKey()

imagecopy[b,a:c] =(0,255,0)
imagecopy[b:d,a] =(0,255,0)
imagecopy[b:d,c] =(0,255,0)
imagecopy[d,a:c] =(0,255,0)

cv2.imshow('Image',imagecopy)
cv2.waitKey()

