# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 19:37:50 2021

@author: gs935
"""


from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract
from gtts import gTTS 
import os
from texttospeech import ttos

results=[]
textRecongized=""

net = cv2.dnn.readNet("frozen_east_text_detection.pb")
pytesseract.pytesseract.tesseract_cmd="C:\Program Files\Tesseract-OCR\\tesseract.exe"

def text_detector(image):
	orig = image.copy()
	(H, W) = image.shape[:2]

	(newW, newH) = (320, 320)
	rW = W / float(newW)
	rH = H / float(newH)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)

	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		boundary = 2

		roi = orig[startY-boundary:endY+boundary, startX - boundary:endX + boundary]
		configuration = ("-l eng --oem 1 --psm 8")

		#text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
  
	# 	inputImage = text
    
    # # Get local maximum:
	# 	kernelSize = 5
	# 	maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
	# 	localMax = cv2.morphologyEx(inputImage, cv2.MORPH_CLOSE, maxKernel, None, None, 1, cv2.BORDER_REFLECT101)
		
	# 	# Perform gain division
	# 	gainDivision = np.where(localMax == 0, 0, (inputImage/localMax))
		
	# 	# Clip the values to [0,255]
	# 	gainDivision = np.clip((255 * gainDivision), 0, 255)
		
	# 	# Convert the mat type from float to uint8:
	# 	gainDivision = gainDivision.astype("uint8")
		
		
	# 	# In[6]:
		
		
	# 	#cv2.imshow("Gain Division", gainDivision)
	# 	#cv2.imwrite("Gain Division.png", gainDivision)
	# 	#cv2.waitKey(0)
	# 	#cv2.destroyAllWindows()
		
		
	# 	# ## OTSU's Thresholding 
		
	# 	# In[7]:
		
		
	# 	grayscaleImage = cv2.cvtColor(gainDivision, cv2.COLOR_BGR2GRAY)
		
	# 	_, binaryImage = cv2.threshold(grayscaleImage, 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		
		
	# 	# In[8]:
		
		
	# 	#cv2.imshow("OTSU's Thresholding", binaryImage)
	# 	text = pytesseract.image_to_string(binaryImage)
	# # cv2.imwrite("OTSU's result.png", binaryImage)
	# 	print(text+"\n")
	# 	#cv2.waitKey(0)
	# 	#cv2.destroyAllWindows()
		
		
	# 	# ## Background Color Filling
		
	# 	# In[9]:
		
		
	# 	kernelSize=3
	# 	opInterations=1
		
	# 	morphkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize,kernelSize))
		
		
	# 	binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphkernel, None, None, opInterations, cv2.BORDER_REFLECT101)
		
		
	# 	# In[10]:
		
		
	# 	#cv2.imshow("Background Color filling", binaryImage)
	# 	#cv2.imwrite("Background Color filling.png", binaryImage)
	# 	text = pytesseract.image_to_string(binaryImage)
	# 	print(text)
	# 	#cv2.waitKey(0)
	# 	#cv2.destroyAllWindows()
		
		
	# 	# ## Flood Filling
		
	# 	# In[11]:
		
		
	# 	cv2.floodFill(binaryImage, mask=None, seedPoint=(int(0), int(0)), newVal=(255))
		
		
	# 	# In[12]:
		
		# retval, img = cv2.threshold(roi,200,255, cv2.THRESH_BINARY)
		# img = cv2.resize(img,(0,0),fx=3,fy=3)
		# img = cv2.GaussianBlur(img,(11,11),0)
		# img = cv2.medianBlur(img,9)
	# 	#cv2.imshow("Flood Fill", binaryImage)
	# 	#cv2.imwrite("flood filling.png", binaryImage)
	# 	#text = pytesseract.image_to_string(binaryImage)
		textRecongized = pytesseract.image_to_string(roi,lang='eng', config=configuration);results.append(textRecongized);

		#textRecongized = pytesseract.image_to_string(binaryImage);print(textRecongized)
        
         
        
        
       
      
  
    # Play the converted file 
        
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
		orig = cv2.putText(orig, textRecongized, (endX,endY+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        
        
	return orig

def callOCR(img):
	# image0 = cv2.imread('stop.jpg')
	# test2=cv2.imread("speed.jpg")
	# test3=cv2.imread("50.jpg")
	# test4=cv2.imread("wait.png")

	# img3=cv2.imread("sign6.png")

	array = [img]

	for i in range(0,1):
		for img in array:
			imageO = cv2.resize(img, (640,320), interpolation = cv2.INTER_AREA)
			orig = cv2.resize(img, (320,320), interpolation = cv2.INTER_AREA)
			textDetected = text_detector(imageO)
			cv2.imshow("Orig Image",orig)
			cv2.imshow("Text Detection", textDetected)
			time.sleep(2)
			k = cv2.waitKey(30)
			if k == 27:
				break
	
