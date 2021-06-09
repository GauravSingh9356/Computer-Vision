# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:56:23 2021

@author: gs9356
"""
import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

img_width, img_height=200,200

model=load_model(os.path.join("./predictor.tf"), "saved_model.pb")



face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv.VideoCapture("videos/crowd_2.mp4")

img_count_full=0

font = cv.FONT_HERSHEY_SIMPLEX
org=(1,15)
class_label=" "
fontScale=1
color=(0,255,0)
thickness=2
pred=0

cap.set(3,480)
cap.set(4,640)

while True:
    img_count_full += 1
    img =cv.imread("mask.png",1)
    
  
    
    scale = 50
    width=int(img.shape[1]*scale/100)
    height = int(img.shape[0]*scale /100)
    dim = (width, height)
    
    color_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
       
    
    faces = face_cascade.detectMultiScale(gray_img, 1.2,2)
    
    img_count=0
    for (x,y,w,h) in faces:
        org = (x+5,y+30)
        img_count+=1
        color_face = color_img[y:y+h, x:x+w]
        cv.imwrite("detected_faces/%d%dface.jpg"%(img_count_full, img_count), color_face)
        img = image.load_img("detected_faces/%d%dface.jpg"%(img_count_full, img_count), target_size=(img_width, img_height))
        
        img = image.img_to_array(img)/255
        img = np.expand_dims(img, axis=0)
        pred_prob = model.predict(img)
        
        pred = np.argmax(pred_prob)
        
        if(pred==0):
            print("User with mask - prediction = ", pred_prob[0][0])
            class_label="Mask"
            cv.imwrite("detected_faces/with_mask/%d%dface.jpg"%(img_count_full, img_count), color_face)
            color=(0,255,0)
            pred= pred_prob[0][0]
        
        else:
            print("User without mask - prediction = ", pred_prob[0][1])
            class_label="Without Mask"
            cv.imwrite("detected_faces/without_mask/%d%dface.jpg"%(img_count_full, img_count), color_face)
            color=(0,0,255)
            pred_prob[0][1]
        
        
        cv.rectangle(color_img, (x,y), (x+w, y+h),color, 3)
        cv.putText(color_img, class_label+" "+str(pred),org, font, fontScale, color, thickness, cv.LINE_AA)
        
        
        cv.imshow("LIVE FACE MASK DETECTION", color_img)
        cv.imwrite("Final.jpg", color_img)
        if(cv.waitKey(1) & 0xFF==ord('q')):    
            break


        
        
        
        
        
cap.release()
cv.destroyAllWindows()

        
        
        
        
        
        
        
        
        
        
        
        
        
    
