import cv2 as cv
import numpy as np

#circles = cv.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])

aimage = cv.imread("data/smarties.png")
image = aimage.copy()
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3, 3), 0)
circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, 20, param1=60, param2=40, minRadius=1, maxRadius=100)

if circles is not None:
    detected_circles = np.uint16(np.around(circles)) # convert numbers in numpy array from float to int
    # int16 ==> commonly used to represent image coordinates.

    for (x, y, r) in detected_circles[0, :]:
        cv.circle(image, (x, y), r, (0, 255, 0), 2)

    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("No circles detected.")
