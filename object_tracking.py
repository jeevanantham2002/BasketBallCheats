import cv2
import imutils
import numpy as np
import argparse
import imutils
import time
################################################################
from centroidtracker import CentroidTracker

path = 'haarcascades/haarcascade_frontalface_default.xml'  # PATH OF THE CASCADE
cameraNo = 1                       # CAMERA NUMBER
objectName = 'Face'       # OBJECT NAME TO DISPLAY
frameWidth= 1920                     # DISPLAY WIDTH
frameHeight = 1080                  # DISPLAY HEIGHT
color= (255,0,255)
#################################################################


cap = cv2.VideoCapture(cameraNo)
# cap.set(3,frameWidth)
# cap.set(4,frameHeight)
def empty(a):
    pass

# CREATE TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result",frameWidth,frameHeight+100)
cv2.createTrackbar("Scale","Result",400,1000,empty)
cv2.createTrackbar("Neig","Result",8,50,empty)
cv2.createTrackbar("Min Area","Result",0,100000,empty)
cv2.createTrackbar("Brightness","Result",180,255,empty)

# LOAD THE CLASSIFIERS DOWNLOADED
cascade = cv2.CascadeClassifier(path)
ct = CentroidTracker()
(H, W) = (None, None)

while True:
    # SET CAMERA BRIGHTNESS FROM TRACKBAR VALUE
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, cameraBrightness)

    # GET CAMERA IMAGE AND CONVERT TO GRAYSCALE
    success,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Centroid Detection Applicaiton
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # print(gray)

    # DETECT THE OBJECT USING THE CASCADE
    scaleVal =1 + (cv2.getTrackbarPos("Scale", "Result") /1000)
    neig=cv2.getTrackbarPos("Neig", "Result")
    detections, rejectLevels, levelWeights = cascade.detectMultiScale3(frame, scaleFactor=1.25,minNeighbors=6, minSize=(30, 30),outputRejectLevels=1)
    print("level weights")
    print(levelWeights)
    # detections = cascade.detectMultiScale(frame,scaleVal, neig,(100,100),True)
    rects=[];
    # print(detections)


    # DISPLAY THE DETECTED OBJECTS
    for i in range(len(detections)):
        (x, y, w, h) = detections[i]
        area = w*h
        minArea = cv2.getTrackbarPos("Min Area", "Result")
        if area >minArea and levelWeights[i]>2.5:

            #Check what Rectangle takes as an input
            box = np.array([x, y, x+w, y+h])
            rects.append(box.astype("int"))

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            roi_color = frame[y:y+h, x:x+w]

    if(rects != None):
        objects = ct.update(rects)
        # print(rects)
        # print(objects)

    if(objects != None):
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
cv2.destroyAllWindows()
cap.stop()