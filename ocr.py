import cv2
import numpy as np
import pytesseract

def preprocess(imgfile, noise_removal):
    #read the image
    img = cv2.imread(imgfile)

    if(noise_removal == True):
   
        #converting to grayscale
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #binarizing the image
        th = 120
        ret, bimg = cv2.threshold(gimg, th, 255, cv2.THRESH_BINARY) 
        #using the global threshold function, can update to adaptive one.

        # #noise removal
        kernal = np.ones((1,1), np.uint8)
        image = cv2.dilate(bimg, kernel=kernal, iterations=1)
        image = cv2.erode(image, kernel=kernal, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel=kernal)
        image = cv2.medianBlur(image, ksize=1)
        cv2.imshow('Preprocessed Image', image)
        cv2.waitKey(0)
        return image
    else:
        #converting to grayscale
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #binarizing the image
        th = 120
        ret, bimg = cv2.threshold(gimg, th, 255, cv2.THRESH_BINARY)
        
        cv2.imshow('Preprocessed Image', bimg)
        cv2.waitKey(0)
        return bimg

def extractLabels(imgfile):
    img = preprocess(imgfile, False)
    res = pytesseract.image_to_string(img)
    return res

#testing 

print(extractLabels("test.png"))
