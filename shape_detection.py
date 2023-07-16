import cv2 as cv
import numpy as np


image = cv.imread("data/square.png")
image = cv.resize(image, (500, 700))
grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(grey, (3, 3), 0)
_, thresh = cv.threshold(blur, 240, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv.contourArea, reverse=True)[-1:]
contour = contours[0]

cv.drawContours(image, contours, -1, (0, 255, 0), 2)


p = cv.arcLength(contour, True)
approx = cv.approxPolyDP(contour, 0.02*p, True)
if len(approx) == 3:
    print("Triangle")
elif len(approx) == 4:
    x, y, w, h = cv.boundingRect(approx)
    rat = float(w/h)
    if rat >= 0.95 and rat <= 1.05:
        print("square")
    else:
        print("rectangle")
elif len(approx) == 5:
    print("Pentagon")
elif len(approx) == 6:
    print("Hexagon") 
elif len(approx) == 10:
    print("Star") 
else:
    print("circle")



cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()