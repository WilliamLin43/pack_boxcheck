# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
import imutils
import time


ap = argparse.ArgumentParser() #4
ap.add_argument("-i1", "--image1", required = True, help = "Path to the image") #5
ap.add_argument("-i2", "--image2", required = True, help = "Path to the image") #5
args = vars(ap.parse_args()) #6

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

detector = cv2.xfeatures2d.SURF_create()
matcher = cv2.DescriptorMatcher_create("BruteForce")

imageA = cv2.imread(args["image1"])
imageB = cv2.imread(args["image2"])
imageA = imutils.resize(imageA, width = 600)
imageB = imutils.resize(imageB, width = 600)
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

kpsA = detector.detect(grayA)
kpsB = detector.detect(grayB)
(kpsA, featuresA) = ROOTSIFT(grayA, kpsA)
(kpsB, featuresB) = ROOTSIFT(grayB, kpsB)


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
    #print(ptA)
    #print(ptB)
    cv2.circle(vis,ptA, 5, c, 1)
    cv2.circle(vis,ptB, 5, c, 1)
    #cv2.line(vis, ptA, ptB, c, 1)


cv2.putText(vis,'Check_Value:'+str(round(((len(matches)/len(rawMatches))*100),4))+'%',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
cv2.imshow("Image Check", np.hstack([vis])) #顯示圖片
#time.sleep(0.05) #delay a little bit, if require
cv2.waitKey(0)
    

    