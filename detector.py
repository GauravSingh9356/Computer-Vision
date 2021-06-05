import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(os.path.join("./predictor.tf"), "saved_model.pb")

cap = cv.VideoCapture(0)

if(cap.isOpened()):
    while(True):
        success, img = cap.read()
        img2 = img.copy()
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gau = cv.GaussianBlur(img, (5, 5), 0)

        img_canny = cv.Canny(img, 50, 250)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv.dilate(img_canny, kernel, iterations=1)

        contours, heirarchy = cv.findContours(
            dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        ans = ""
        for cnt in contours:
            area = cv.contourArea(cnt)
            if(area > 2500):
                peri = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02*peri, True)
                x, y, w, h = cv.boundingRect(approx)
                cv.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 3)
                img_new = img_gray[y:y+h, x:x+w]
                img_new = cv.resize(img_new, (28, 28),
                                    interpolation=cv.INTER_AREA)
                img_new = tf.keras.utils.normalize(img_new, axis=1)
                img_new = np.array(img_new).reshape(1, 28, 28, 1)

                y_pred = model.predict(img_new)
                ans = np.argmax(y_pred)

        y_p = str('Predicted Value is '+str(ans))
        cv.putText(img2, y_p, (30, 50), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("Frame", img2)
        cv.imshow("Contours Frame", img_canny)
        cv.imwrite("Canny.png", img_canny)

        key = cv.waitKey(1)
        if key == 27:
            break


cap.release()
cv.destroyAllWindows()
