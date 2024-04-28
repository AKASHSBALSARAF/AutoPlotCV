import cv2
import numpy as np
import statistics as st
import pytesseract
from pytesseract import Output
from preprocess import detectAxesAndLengthScales as le 
import warnings

#ignores warnings
warnings.filterwarnings('ignore') 

def enhance(img):
    img = cv2.resize(img,(0,0),fx=3,fy=3)
    gimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bimg = cv2.threshold(gimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cv2.imshow("Axis", bimg)
    cv2.waitKey()    
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
    try:
        divx = x[1:]/x[0:-1]
        divy = y[1:]/y[0:-1]
    except ZeroDivisionError:
        pass
    ll['v'][0], ll['c'][0] = np.unique(diffx, return_counts=True)
    ll['v'][1], ll['c'][1] = np.unique(diffy, return_counts=True)
    ll['v'][2], ll['c'][2] = np.unique(divx, return_counts=True)
    ll['v'][3], ll['c'][3] = np.unique(divy, return_counts=True)

    #for x axis
    if len(ll['v'][0])<=len(diffx)-1:
        print('Linear distribution found on X-axis')
        typex = 'lin'
        dx = st.mode(diffx)
    elif len(ll['v'][2])<=len(divx)-1:
        print('Logarithmic distribution found on X-axis')
        typex = 'log'
        dx = st.mode(divx)
    #for y axis
    if len(ll['v'][1])<=len(diffy)-1:
        print('Linear distribution found on Y-axis')
        typey = 'lin'
        dy = st.mode(diffy)
    elif len(ll['v'][3])<=len(divy)-1:
        print('Logarithmic distribution found on Y-axis')
        typey = 'log'
        dy = st.mode(divy)

    return dx,dy,typex,typey

def extractLabels(imgfile):
    img = cv2.imread(imgfile)
    [graph, nongraph, b, d, c, a, lengthscaley, lengthscalex] = le(img)
    nongraph[b-5:d+5,c-5:a+5] = 255
    cv2.imwrite('nongraph.jpg', nongraph)
    #running OCR on vertical axis
    ngv = np.copy(nongraph[:,:c-3])
    ngv = enhance(ngv)
    cv2.imwrite('ngv.jpg', ngv)
    dic = pytesseract.image_to_data(ngv, config='--psm 4',output_type=Output.DICT)
    n_boxes = len(dic['level'])
    resv = []
    for i in range(n_boxes):
        if is_number(dic['text'][i]): 
            resv.append(float(dic['text'][i]))
    #running OCR on horizontal axis
    ngh = np.copy(nongraph[d+5:,:])
    ngh = enhance(ngh)
    cv2.imwrite('ngh.jpg', ngh)
    dic = pytesseract.image_to_data(ngh, config='--psm 4', output_type=Output.DICT)
    n_boxes = len(dic['level'])
    resh = []
    for i in range(n_boxes):
        if is_number(dic['text'][i]): 
            resh.append(float(dic['text'][i]))

    resv = np.flip(resv)
    print(resh,resv)
    [dx,dy,typex,typey]=linORlog(resh, resv)
    print([dx,dy,typex,typey])


extractLabels("testImages/testexp.jpg")
