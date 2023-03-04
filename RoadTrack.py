import cv2

img = cv2.imread("./data/samples/roadSample.jpeg")

grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(grayScale, 0, 200)

cv2.imshow("img", img)
cv2.waitKey(0)
