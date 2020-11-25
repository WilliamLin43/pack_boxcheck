# -*- coding: utf-8 -*-
import cv2 #load opencv
import time
import numpy as np
import argparse
from VideoStream import VideoStream#, VideoSave
from numpy import genfromtxt
import configparser
import os
#import imutils


'''
#set raspberry pi GPIO
import RPi.GPIO as gpio
gpio.setmode(gpio.BCM)
gpio.setup(14,gpio.OUT)
'''

#Log Flag
LOG_FLAG=0

#cut image sub program
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

#ROOTSIFT define
def ROOTSIFT(grayIMG, kpsData):
    extractor = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = extractor.compute(grayIMG, kpsData)

    if len(kps) > 0:
        #L1-正規化
        eps=1e-7
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        #取平方根
        descs = np.sqrt(descs)
        return (kps, descs)
    else:
        return ([], None)

#camera setting
parser = argparse.ArgumentParser()
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720') #1280x720;1920x1080
parser.add_argument('--fps', help='Frames per second',
                    default=15)

args = parser.parse_args()
resW, resH = args.resolution.split('x')
w, h = int(resW), int(resH)
FPS = int(args.fps)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(w,h),framerate=FPS).start()
out=None

time.sleep(2)

#read images from test video
#camera = cv2.VideoCapture(1) 

while(True):
    try:
        folderpath = "./UNPK01_PP_BOX/" # Check folder exist or not.
        print(folderpath)
        if os.path.isdir(folderpath):
            print(" Input folder exist.")
            files= os.listdir(folderpath) #得到資料夾下的所有檔名稱
            #print(files)
            for file in files: #遍歷資料夾
                filename = cv2.imread(folderpath+'/'+file)
                img=filename
        
                # Define and parse input arguments
                config=configparser.ConfigParser()
                config.read("config.txt")    
                Set_VALUE1 = float(config["DEFAULT"]["VALUE1"])#args. detection value
                Set_VALUE2 = float(config["DEFAULT"]["VALUE2"])#args. detection value
            
                t1 = cv2.getTickCount()
                #read images from camera
                #frame = videostream.read()
                #frame = cv2.flip(frame, 0) #Rotate image
                '''
                #read images from test video
                ret, frame = camera.read()
                if not ret:
                    break
                
                img=frame
                '''
                
                #read points record from file
                dataPath=r'./cutpoints.csv'
                points=genfromtxt(dataPath,delimiter=',').astype(int).tolist()
                #img=cv2.imread("./UNPK01_PP_BOX/PP BOX NG-001.png")
                #cv2.imshow("Image Check", np.vstack([img])) #顯示圖片
                #cv2.waitKey(0)
                
                #cut image & save log
                CheckArea=croppoly(img,points)
                cv2.imwrite("./cut.png", CheckArea)
                #cv2.imshow("Image Check", np.vstack([CheckArea])) #顯示圖片
                #cv2.waitKey(0)
                
            
                detector = cv2.xfeatures2d.SURF_create()
                matcher = cv2.DescriptorMatcher_create("BruteForce")
                
                imageA = cv2.imread("./cut0-1.png")
                imageB = CheckArea
                #imageB = cv2.imread("./cut.png")
                #imageA = imutils.resize(imageA, width = 600)
                #imageB = imutils.resize(imageB, width = 600)
                grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
                
                kpsA = detector.detect(grayA)
                kpsB = detector.detect(grayB)
                (kpsA, featuresA) = ROOTSIFT(grayA, kpsA)
                (kpsB, featuresB) = ROOTSIFT(grayB, kpsB)
                
                print(len(featuresA))
                
                print(len(featuresB))
                
                
                rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
                matches = []
                                 
                print(len(rawMatches))
                
                
                for m in rawMatches:
                    #print ("#1:{} , #2:{}".format(m[0].distance, m[1].distance))
                    if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                        matches.append((m[0].trainIdx, m[0].queryIdx))
                        
                (hA, wA) = imageA.shape[:2]
                (hB, wB) = imageB.shape[:2]
                vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
                vis[0:hA, 0:wA] = imageA
                vis[0:hB, wA:] = imageB
                
                print(len(matches))
                
                for (trainIdx, queryIdx) in matches:
                    color = np.random.randint(0, high=255, size=(3,),dtype='int')
                    #print(color)
                    c = tuple(map(int, color))
                    ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
                    ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
                    print(ptA)
                    print(ptB)
                    cv2.circle(vis,ptA, 5, c, 1)
                    cv2.circle(vis,ptB, 5, c, 1)
                    #cv2.line(vis, ptA, ptB, c, 1)
                
                Value1=round(((len(matches)/len(rawMatches))*100),4)
                cv2.putText(vis,'Check_Value:'+str(Value1)+'%',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                
                #time.sleep(0.05) #delay a little bit, if require
                #cv2.waitKey(0)
                
                frameDelta = cv2.absdiff(imageA, imageB)
                #frameDelta = cv2.subtract(imageA, imageB)
                #thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
                #thresh = cv2.dilate(thresh, None, iterations=2)
                #thresh = cv2.erode(thresh, None, iterations=2)
                vis2 = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
                vis2[0:hA, 0:wA] = imageA
                vis2[0:hB, wA:] = frameDelta
                
                Value2=round(((1-(np.sum(frameDelta)/np.sum(imageA)))*100),4)
                cv2.putText(vis2,'Check_Value:'+str(Value2)+'%',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                
                
                #check setting value is over or not
                if Set_VALUE1 < Value1 or Set_VALUE2 < Value2:        
                    BOX_ON=True
                else:
                    cv2.putText(vis,'Below Setting Value:'+str(Set_VALUE1)+'%',(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                    cv2.putText(vis2,'Below Setting Value:'+str(Set_VALUE2)+'%',(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                    LOG_FLAG=0
                    BOX_ON=False
                        
                cv2.imshow("Image Check", np.vstack([vis,vis2])) #顯示圖片
                #cv2.waitKey(0)
                
                if BOX_ON:
                    #gpio.output(14,gpio.LOW)
                    #print("Check ok")
                    LOG_FLAG=0            
                else:
                    #gpio.output(14,gpio.HIGH)
                    #os.system("mpg321 ./colorincorrect.mp3 &") #paly warning audio message
                    #time.sleep(3) #waiting for play finish
                    if LOG_FLAG == 0:
                        fp=open("./Errorlog.csv", "a")
                        #Text=time.strftime("%Y%m%d%H%M%S", time.localtime())+",Over Setting,Value1 Setting:"+str(Set_VALUE1)+",Value1 Result:"+str(Value1)+",Value2 Setting:"+str(Set_VALUE2)+",Value2 Result:"+str(Value2)
                        Text=time.strftime("%Y%m%d%H%M%S", time.localtime())+","+str(file)+","+str(Set_VALUE1)+","+str(Value1)+","+str(Set_VALUE2)+","+str(Value2)
                        fp.write(Text+"\n")
                        LOG_FLAG=1
                        fp.close()                            
                
                time.sleep(0.2) #delay a little bit, if require

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
    except Exception as err:        
        print("except:"+str(err))
        break

cv2.destroyAllWindows()
videostream.stop()
#gpio.output(14,gpio.LOW)
    