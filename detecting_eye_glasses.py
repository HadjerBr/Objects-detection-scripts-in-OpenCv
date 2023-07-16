from signal import CTRL_C_EVENT
import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
glass_detector = cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

cap = cv.VideoCapture(0)

while(True):
    _, frame = cap.read() # ret is a Boolean variable that indicates the success of reading a frame from the video capture source.
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        mgray = gray[y:y+h, x:x+w] # (y, x)
        mframe = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(mgray)
        glasses = glass_detector.detectMultiScale(mgray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(mframe, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        for (gx, gy, gw, gh) in glasses:
            cv.rectangle(mframe, (gx, gy), (gx+gw, gy+gh), (0, 0, 255), 3)


    
    cv.imshow("video", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()



