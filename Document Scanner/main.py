import cv2 as cv
import numpy as np


widthImg = 840
heightImg = 680

frameWidth = 640
frameHeight = 480
cap = cv.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 150)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv.cvtColor(
                        imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContors(img):
    biggest = np.array([])
    maxArea = 0
    contors, heirarchy = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contors:
        area = cv.contourArea(cnt)
        if(area > 5000):
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            print(approx)
            if(area > maxArea and len(approx) == 4):
                area = maxArea
                biggest = approx

    cv.drawContours(imgContor, biggest, -1, (0, 255, 0), 50)

    print(biggest)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    #print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # print("NewPoints",myPointsNew)
    return myPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32(
        [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv.warpPerspective(img, matrix, (widthImg, heightImg))
    imgCropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    imgCropped = cv.resize(imgCropped, (widthImg, heightImg))

    return imgCropped


def preProcessing(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDilation = cv.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv.erode(imgDilation, kernel, iterations=1)
    return imgThres


# while(True):
# success, img = cap.read()
img = cv.imread("test.jpg", 1)
img = cv.resize(img, (widthImg, heightImg))

# img = img[60:, :]

cv.imshow("Original", img)
imgContor = img.copy()
imgThres = preProcessing(img)
cv.imshow("ImgThres", imgThres)

biggest = getContors(imgThres)
if (biggest.size != 0):
    cv.imshow("ImgContor", imgContor)
    imgWarped = getWarp(img, biggest)
    cv.imshow("Result", imgWarped)

    cv.imwrite("ImgContor.jpg", imgContor)
    imgWarped = getWarp(img, biggest)
    cv.imwrite("Result.jpg", imgWarped)
    cv.imwrite("canny.jpg", imgThres)

    imageArray = ([img, imgThres, imgContor, imgWarped])
else:
    imageArray = ([imgContor, img])

stackedImages = stackImages(0.4, imageArray)
cv.imshow("WorkFlow", stackedImages)
cv.imwrite("Combined Process.jpg", stackedImages)


cv.waitKey()

# if(cv.waitKey(1) & 0xFF == ord('q')):
#     break
