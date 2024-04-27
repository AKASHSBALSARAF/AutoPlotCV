import cv2
import numpy as np
import statistics as st
import pytesseract
from pytesseract import Output
from lengthscalesAndEdges import lengthscalesAndEdge as le 

def enhance(img):
    img = cv2.resize(img,(0,0),fx=3,fy=3)
    gimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bimg = cv2.threshold(gimg, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]    
    return bimg

def is_number(string):
  try:
    float(string)
    return True  # True if string is a number contains a dot
  except ValueError:  # String is not a number
    return False

def linORlog(x,y):
    #finding whether the linear or logarithmic
    #logic: if difference is constant than linear else if the multiple is constant logarithmic
    ll = {'v':[0,1,2,3],'c':[0,1,2,3]}
    x = np.array(x)
    y = np.array(y)
    diffx = np.round(np.diff(x),4)
    diffy = np.round(np.diff(y),4)
    divx = x[2:]/x[1:-1]
    divy = y[2:]/y[1:-1]
    ll['v'][0], ll['c'][0] = np.unique(diffx, return_counts=True)
    ll['v'][1], ll['c'][1] = np.unique(diffy, return_counts=True)
    ll['v'][2], ll['c'][2] = np.unique(divx, return_counts=True)
    ll['v'][3], ll['c'][3] = np.unique(divy, return_counts=True)

    #for x axis
    if len(ll['v'][0])<=len(diffx):
        print('Linear distribution found on X-axis')
        typex = 'lin'
        dx = st.mode(diffx)
    elif len(ll['v'][2])<=len(divx):
        print('Logarithmic distribution found on X-axis')
        typex = 'log'
        dx = st.mode(divx)
    #for y axis
    if len(ll['v'][1])<=len(diffy):
        print('Linear distribution found on Y-axis')
        typey = 'lin'
        dy = st.mode(diffy)
    elif len(ll['v'][3])<=len(divy):
        print('Logarithmic distribution found on Y-axis')
        typey = 'log'
        dy = st.mode(divy)

    return dx,dy,typex,typey

    
def extractLabels(imgfile):
    img = cv2.imread(imgfile)
    [graph, nongraph, a, b, c, d, lengthscaley, lengthscalex] = le(img)
    nongraph[b-10:d+10,a-10:c+10] = 255
    
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

    resv = np.flip(resv)
    [dx,dy,typex,typey]=linORlog(resh, resv)
    print([dx,dy,typex,typey])
    return [resh, resv]
#testing 

print(extractLabels("testImages/test2.png"))
