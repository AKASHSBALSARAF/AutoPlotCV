import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from lengthscalesAndEdges import lengthscalesAndEdge as le 

def enhance(img):
    img = cv2.resize(img,(0,0),fx=3,fy=3)
    gimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bimg = cv2.threshold(gimg, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cv2.imshow('enhanced image', bimg)
    cv2.waitKey(0)    
    return bimg

def is_number(string):
  try:
    float(string)
    return True  # True if string is a number contains a dot
  except ValueError:  # String is not a number
    return False

def extractLabels(imgfile):
    img = cv2.imread(imgfile)
    [graph, nongraph, a, b, c, d, lengthscaley, lengthscalex] = le(img)
    nongraph[b-10:d+10,a-10:c+10] = 255
    cv2.imshow('Non-graph', nongraph)
    cv2.waitKey(0)
    
    #running OCR on vertical axis
    ngv = np.copy(nongraph[:,:(a-10)])
    ngv = enhance(ngv)
    dic = pytesseract.image_to_data(ngv, config='--psm 4',output_type=Output.DICT)
    n_boxes = len(dic['level'])
    resv = []
    for i in range(n_boxes):
        if is_number(dic['text'][i]): 
            resv.append(float(dic['text'][i]))

    #running OCR on horizontal axis
    ngh = np.copy(nongraph[(d+10):,:])
    ngh = enhance(ngh)
    dic = pytesseract.image_to_data(ngh, config='--psm 4', output_type=Output.DICT)
    n_boxes = len(dic['level'])
    resh = []
    for i in range(n_boxes):
        if is_number(dic['text'][i]): 
            resh.append(float(dic['text'][i]))
    
    return [resh, resv]
#testing 

print(extractLabels("testImages/test2.png"))
