import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while True:
    if frame1 is None or frame2 is None:
        break

    if frame1.shape == frame2.shape:
        diff = cv.absdiff(frame1, frame2)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3, 3), 0)
        ret, thr = cv.threshold(blur, 25, 255, cv.THRESH_BINARY)
        dilate = cv.dilate(thr, None, iterations=2)
        contours, _ = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        
        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour) # to get x y w and h of the contour
            if cv.contourArea(contour) < 10000:
                continue
            cv.rectangle(frame1, (x,y), (x+w, y+h), (0, 0, 255), 2)

    cv.imshow("Motion Detection", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv.waitKey(40) == 27: 
        break

cv.destroyAllWindows()
cap.release()
