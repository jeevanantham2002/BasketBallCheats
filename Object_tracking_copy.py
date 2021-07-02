from collections import deque
import numpy as np
import cv2
import imutils

################################################################
path = 'haarcascade_frontalface_default.xml'  # PATH OF THE CASCADE
cameraNo = 0  # CAMERA NUMBER
objectName = 'Face'  # OBJECT NAME TO DISPLAY
frameWidth = 1920  # DISPLAY WIDTH
frameHeight = 1080  # DISPLAY HEIGHT
color = (255, 0, 255)
deque_len = 10;
#################################################################


cap = cv2.VideoCapture(cameraNo)


# cap.set(3,frameWidth)
# cap.set(4,frameHeight)
def empty(a):
    pass


# CREATE TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result", frameWidth, frameHeight + 100)
cv2.createTrackbar("Scale", "Result", 400, 1000, empty)
cv2.createTrackbar("Neig", "Result", 8, 50, empty)
cv2.createTrackbar("Min Area", "Result", 0, 100000, empty)
cv2.createTrackbar("Brightness", "Result", 180, 255, empty)

# LOAD THE CLASSIFIERS DOWNLOADED
cascade = cv2.CascadeClassifier(path)
pts = deque(maxlen=deque_len)

while True:
    # SET CAMERA BRIGHTNESS FROM TRACKBAR VALUE
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, cameraBrightness)

    # GET CAMERA IMAGE AND CONVERT TO GRAYSCALE
    success, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DETECT THE OBJECT USING THE CASCADE
    scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Result") / 1000)
    neig = cv2.getTrackbarPos("Neig", "Result")
    objects, rejectLevels, levelWeights = cascade.detectMultiScale3(frame, scaleFactor=1.25,minNeighbors=6, minSize=(30, 30),outputRejectLevels=1)

    # DISPLAY THE DETECTED OBJECTS
    for i in range(len(objects)):
        (x, y, w, h) = objects[i]
        area = w * h
        minArea = cv2.getTrackbarPos("Min Area", "Result")
        if area > minArea  and levelWeights[i]>2.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, objectName, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

            x_coordinate = int((x+(w/2)))
            y_coordinate = int((y+(h/2)))
            center = (x_coordinate,y_coordinate)
            pts.appendleft(center)

            cv2.circle(frame, center, 4, (0, 255, 0), -1)
            roi_color = frame[y:y + h, x:x + w]

    #Get past points from deque
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(deque_len / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break