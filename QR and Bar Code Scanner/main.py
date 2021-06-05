import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode


# web cam code
# cap = cv.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

img = cv.imread("2.png", 1)

for barcode in decode(img):
    myData = barcode.data.decode("utf-8")
    print(myData)
    print(barcode.polygon)
    pts = np.array([barcode.polygon], np.int32)
    pts = pts.reshape((4, 1, 2))
    print(pts.shape)
    cv.polylines(img, [pts], True, (255, 0, 255), 5)
    pts2 = barcode.rect
    cv.putText(img, myData, (pts2[0], pts2[1]),
               cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)


cv.imshow("Result", img)
cv.waitKey(0)


cv.destroyAllWindows()
