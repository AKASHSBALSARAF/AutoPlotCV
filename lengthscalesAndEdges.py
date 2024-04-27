# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:52:22 2024

@author: Pratham Gupta
"""
#This is for testing of filters and basic image processing

import cv2 
import numpy as np

def lengthscalesAndEdge(load):
    loadcopy = load

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
    #print(column_sum)
    #print(max(column_sum))
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
    #print(row_sum)
    #print(max(row_sum))
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

    #print(b,d,c,a)

    offset[b,a:c] =128
    offset[b:d,a] =128
    offset[b:d,c] =128
    offset[d,a:c] =128
    # cv2.imshow('Offset', offset)
    # cv2.waitKey()

    imagecopy[b,a:c] =(0,255,0)
    imagecopy[d,a:c] =(0,255,0)
    imagecopy[b:d,a] =(0,255,0)
    imagecopy[b:d,c] =(0,255,0)

    # cv2.imshow('Image',imagecopy)
    # cv2.waitKey()

    graph = imagecopy[b-3:d+3,a-3:c+3]
    nongraph = imagecopy
    nongraph[b:d,a:c] = 255
    graphcopy = graph

    # Display cropped image
    # cv2.imshow("Cropped", graph)
    # cv2.waitKey()

    # Save the cropped image
    #cv2.imwrite("testImages/Cropped Image.jpg", cropped_image)

    graygraph= cv2.cvtColor(graph, cv2.COLOR_BGR2GRAY)
    graygraphcopy = cv2.cvtColor(graphcopy,cv2.COLOR_BGR2GRAY)
    graygraph = np.float32(graygraph)

    dst = cv2.cornerHarris(graygraph,2,3,0.05)
    # Look into Shi-Tomasi corner detection
    # Threshold for an optimal value, it may vary depending on the image.
    graygraph[dst>0.06*dst.max()]=255

    corneronly = graygraph-graygraphcopy

    corneronly[:-9,9:]=0
    #cv2.imshow('Corneronly', corneronly)
    #cv2.waitKey()

    (var1,var2) = np.shape(corneronly)
    indexhighlow= []
    indexlowhigh= []
    for i in range(var2-1):
     for j in range(var1-1):
        if corneronly[j,i] > 128:
            if corneronly[j+1,i] < 128:
                indexhighlow.append((j,i))
        elif corneronly[j,i] < 128:
            if corneronly[j+1,i] > 128:
                indexlowhigh.append((j+1,i))

    second_element_count = {}

    for tup in indexlowhigh:
     second_element = tup[1]
     if second_element in second_element_count:
        second_element_count[second_element] += 1
     else:
        second_element_count[second_element] = 1

    max_occurrence = max(second_element_count.values())
    max_occurrence_elements = [k for k, v in second_element_count.items() if v == max_occurrence]
    maximum = max_occurrence_elements[0]
    filtered_array = [tup for tup in indexlowhigh if tup[1] == maximum]
    print(filtered_array)

    differences = []
    for i in range(1, len(filtered_array)):
     difference = filtered_array[i][0] - filtered_array[i - 1][0]
     differences.append(difference)
    print(differences)
    lengthscaley = np.ceil(np.mean(differences))

    for i in range(var2-1):
     for j in range(var1-1):
        if corneronly[j,i] > 128:
            if corneronly[j,i+1] < 128:
                indexhighlow.append((j,i))
        elif corneronly[j,i] < 128:
            if corneronly[j,i+1] > 128:
                indexlowhigh.append((j,i+1))

    first_element_count = {}

    for tup in indexlowhigh:
     first_element = tup[0]
     if first_element in first_element_count:
        first_element_count[first_element] += 1
     else:
        first_element_count[first_element] = 1

    max_occurrence = max(first_element_count.values())
    max_occurrence_elements = [k for k, v in first_element_count.items() if v == max_occurrence]
    maximum = max_occurrence_elements[0]
    filtered_array = [tup for tup in indexlowhigh if tup[0] == maximum]
    print(filtered_array)

    # differences = []
    # for i in range(1, len(filtered_array)):
    #     difference = filtered_array[i][1] - filtered_array[i - 1][1]
    #     differences.append(difference)
    # print(differences)
    # for i in range(len(differences)):
    #     if differences[i-1] < 10:
    #         differences.pop(i-1)
    # print(differences)
    # lengthscale2 = np.ceil(np.mean(differences))
    # print(lengthscale2)

    i = np.arange(1,len(filtered_array))
    filtered_array=np.array(filtered_array)
    diff = filtered_array[i]-filtered_array[i-1]
    diff = diff[np.where(diff>5)]
    lengthscalex = np.ceil(np.mean(diff))
    
    return(graph,nongraph,a,b,c,d,lengthscaley,lengthscalex)