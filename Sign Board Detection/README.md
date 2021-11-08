# Sign Board Detection

Train and classify Traffic Signs using Convolutional neural networks. This will be done using OPENCV in real time using a simple webcam

## Convolutional Neural Network
<img src="https://www.mdpi.com/applsci/applsci-10-01245/article_deploy/html/images/applsci-10-01245-g002.png"/>
A Convolutional Neural Network are very similar ordinary neural networks. They are made up of neurons with learnable weights and biases which is also known as supervised learning. All the basic idea learned or used for ordinary neural networks are also applicable to CNNs. The only difference between CNN and the ordinary neural network is that CNN assumes that input is an image rather than a vector. This vastly reduces the number of parameters to be tuned in a model. Convolutional Nets models are easy and faster to train on images comparatively to the traditional models.

## Labels and dataset

In this project we had trained traffic signs with over 35000 images of 43 different classes/labels with the help of tensorflow and keras. These labels include cases such as speed limit, traffic signals, animal crossing , etc. We have taken the dataset from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). This dataset is big enough which will help train model more accurately and help us achieve better results.



## Data Pre-Processing
One of the limitations of the CNN model is that they cannot be trained on a different dimension of images. So, it is mandatory to have same dimension images in the dataset.
We’ll check the dimension of all the images of the dataset so that we can process the images into having similar dimensions. We will transform the image into the required dimension using opencv package.
Same as the traditional neural networks convolutional neural networks is same as a sequence of layers. All the layers transforms an input image to an output image with some differentiable function that might include parameters. The CNN architecture consists of three types of layers: Convolutional Layer, Pooling Layer, and Fully-Connected Layer.



## Training the model
To train the model, we have used an Adam optimizer with batch size 32 and number of epoch is 10.

We followed a simple approach and ran only 10 epochs of the training and observed the validation error trying to set it on minimum level and also due to limitation of computational power. It is very important to consider mainly validation error while improving the model. Decreasing only the error with respect to training data can easily lead to unwanted model over fitting.
Please make sure to update tests as appropriate.

<img src="https://github.com/GauravSingh9356/Computer-Vision/blob/master/Sign%20Board%20Detection/Screenshot%20(408).png"/>

## Traffic sign image using webcam
we are testing our model on a real time video which is captured using laptop web cam or any external desktop camera.

After we get traffic sign images we are processing every frame of the video (up to 30 FPS) and preprocess it first then input each image to our pickled model.

Our model will output the predictions and will show us the class name in which that image belongs to. Our system will also show the probability of the correctness of prediction.

## Result and Accuracy( Image Detection)
We have successfully implemented a Convolutional Neural Network to the Traffic Sign Recognition task with greater than 90% accuracy on average. We have covered how deep learning can be used to classify traffic signs with high accuracy, employing a variety of pre-processing and visualization techniques and trying different model architectures. We built a simple easy to understand CNN model to recognize road traffic signs accurately. Our model reached close to close to 97% accuracy on the test set which isgood considering limitation of computational power and with a fairly simple architecture. 

<img src="https://github.com/GauravSingh9356/Computer-Vision/blob/master/Sign%20Board%20Detection/Screenshot%20(411).png"/>
## VGG16 Transfer Learning Model
Transfer learning generally refers to a process where a model trained on one problem is used in some way on a second related problem. In deep learning, transfer learning is a technique whereby a neural network model is first trained on a problem similar to the problem that is being solved. One or more layers from the trained model are then used in a new model trained on the problem of interest.
* **Utilizing the VGG16 Model :** We implimented VGG16 model and trained the same to detect traffic signals. VGG16 is a convolutional neural network trained on a subset of the ImageNet dataset, a collection of over 14 million images belonging to 22,000 categories.
<img src="https://storage.googleapis.com/lds-media/images/vgg16-architecture.width-1200.jpg"/>
The VGG16 Model has 16 Convolutional and Max Pooling layers, 3 Dense layers for the Fully-Connected layer, and an output layer of 1,000 nodes.

* **Use-Cases and Implementation :** Unfortunately, there are two major drawbacks with VGGNet:
    - It is painfully slow to train.
    - The network architecture weights themselves are quite large (concerning disk/bandwidth).
Due to its depth and number of fully-connected nodes, VGG16 is over 533MB. This makes deploying VGG a tiresome task.

* **Result and Accuracy :** <img src="https://github.com/prashantprem/Computer-Vision/blob/master/Sign%20Board%20Detection/vgg_result.png"/> 
The above image indicates the input of the image and another one when I upload traffic this shows the desired output of an image. Traffic Sign Classification is very necessary to identify traffic signs much faster even though the traffic sign is in any size or shape. Even though the traffic sign is a blur and faded then also with this we can detect it very easily as shown in image(on right). Traffic sign detection is very necessary to take place detection automatically when we are in travelling in a vehicle. This helps to detect the traffic signal before the traffic sign is arrived. This alerts the when the person is in sleepy mode. This also guides the way of road automatically.
**The VGG16 is implemented for the traffic sign classification of images. VGG16 holds 82.77% Accuracy, but as our CNN model has better accuracy we decided to dump this model.**


## Text Detection
Text detection is the process of localizing where an image text is. You can think of text detection as a specialized form of object detection.
In object detection, our goal is to detect and compute the bounding box of all objects in an image and determine the class label for each bounding box while in text detection our goal is to automatically compute the bounding boxes for every region of text in an image. 


* **Tesseract OCR(Text localization and detection) :**  At first for Text Detection part we decided to use Tesseract which includes a highly accurate deep learning-based model for text recognition to read text from images and to implement this using python we used a wrapper function tesseract.py.




* **Image Processing :** Tesseract model is trained on data set images with text  black and background white so we did following image processing:
    * Resize the image with variable height and width(multiply 0.5 and 1 and 2 with image height and width).
    * Convert the image to Gray scale format(Black and white).
    * Remove the noise pixels and make more clear(Filter the image).


* **EAST (Efficient accurate scene text detector) :** As we did not get the satisfactory results using Tesseract all alone we integrated EAST alongwith Tesseract in our model. EAST is a very robust deep learning method for text detection. It can find horizontal and rotated bounding boxes, it can be used in combination with any text recognition method. EAST can also detect text both in images and in the video it runs near real-time at 13FPS on 720p images with high text detection accuracy. Another benefit of this technique is that its implementation is available in OpenCV 3.4.2 and OpenCV 4.
    * Use EAST text detection model to detect the presence of text in an image.
    * Extract the text Region of Interest (ROI) from the image using basic image cropping/NumPy array slicing
    * Take the text ROI, and then pass it into Tesseract to actually OCR the text.
IF CNN model fails to detect images then EAST and Tesseract will detect the image hence our model has become more robust.



* **Text to speach :**  After the text is detected it can also be converted into spoken language with the help of text to speach functionality.





 






