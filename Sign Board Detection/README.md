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

## Accuracy (v1.0.1)
We have successfully implemented a Convolutional Neural Network to the Traffic Sign Recognition task with greater than 90% accuracy on average. We have covered how deep learning can be used to classify traffic signs with high accuracy, employing a variety of pre-processing and visualization techniques and trying different model architectures. We built a simple easy to understand CNN model to recognize road traffic signs accurately. Our model reached close to close to 97% accuracy on the test set which isgood considering limitation of computational power and with a fairly simple architecture. 

