import cv2 as cv
import numpy as np

# Haar Cascade Classifiers
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)


while(True):
    ret, frame = cap.read() # ret is a Boolean variable that indicates the success of reading a frame from the video capture source.
    if ret==True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    cv.imshow("video", frame)

cap.release()
cv.destroyAllWindows()


