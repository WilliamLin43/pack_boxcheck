# -*- coding: utf-8 -*-
import cv2 #載入opencv模組
import numpy as np
import csv


drag_start = None
points=[]
global gray
    
def imgSize(img):
    return tuple(img.shape[1::-1])
def updateImg(gray):
    global points
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pts=np.asarray(points)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,255,255))
    cv2.imshow("gray", img)

def onmouse(event, x, y, flags, params):
    global drag_start,points,height,width
    if event == cv2.EVENT_LBUTTONDOWN:
        points.insert(len(points)-1,[x,y])
        if drag_start == None:
            drag_start=x,y
            points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        #drag_start = None
        points.pop()#pop the last
        updateImg(gray)
    elif drag_start:
        #update last points = mouse position
        points[len(points)-1]=[x,y]
        updateImg(gray)

def crop(img,p):
    #global points
    pts=np.asarray(p)
    pts = pts.reshape((-1,1,2))
    ##Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y:y+h, x:x+w].copy()
    return cropped

def croppoly(img,p):
    #global points
    pts=np.asarray(p)
    pts = pts.reshape((-1,1,2))
    ##Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.fillPoly(mask,pts=[pts],color=(255,255,255))
    #cv2.copyTo(cropped,mask)
    dst=cv2.bitwise_and(cropped,cropped,mask=mask)
    bg=np.ones_like(cropped,np.uint8)*255 #fill the rest with white
    cv2.bitwise_not(bg,bg,mask=mask)
    dst2=bg+dst
    return dst2
    #return mask
 
def selectpoly(img):
    global gray,points
    cv2.namedWindow("gray",1)
    cv2.setMouseCallback("gray", onmouse)
    #img=cv2.imread(img)
    width,height=imgSize(img)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
       
    points.pop()#get rid of the mouse point
    p=points
    print(p)      
       
    #dst2=croppoly(gray,p)
    dst2=croppoly(img,p)
    #points=[]#reset for next crop
    return dst2,p


if __name__ == '__main__':
    filename = './UNPK01_PP_BOX/PP BOX OK-001.bmp'
    filename = cv2.imread(filename)    
    
    img=filename
    
    dst,points=selectpoly(img)
    #points=[[440, 344], [779, 341], [731, 646], [367, 636], [409, 471]]
    #dst=croppoly(img,points)
    cv2.imshow("cut image",dst)
    cv2.imwrite("././cut0-1.png", dst)
    # write points data to file
    with open('./cutpoints.csv', 'w', newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerows(points)
        points=[]
        drag_start = None
    
    
'''
    for i in range(2):
        dst,points=selectpoly(img)
        #points=[[440, 344], [779, 341], [731, 646], [367, 636], [409, 471]]
        #dst=croppoly(img,points)
        cv2.imshow("cut image",dst)
        cv2.imwrite("./cut"+str(i)+"-0.png", dst)
        # write points data to file
        with open('./cutpoints'+str(i)+'.csv', 'w', newline='') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerows(points)
            points=[]
            drag_start = None
'''            
    
cv2.destroyAllWindows()

