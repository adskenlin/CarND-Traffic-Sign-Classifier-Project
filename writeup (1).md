# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization1.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/visualization2.jpg "Visualization"
[image4]: ./examples/new_test_images "new test images"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


### Data Set Summary & Exploration

I used the numpy to calculate summary statistics of the traffic
signs data set:

* The size of original training set is 34799.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
It is bar charts showing the distribution of the data set. 

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale and normalization because it made the input data better fitting on the training model and improved the training quality and speeding up the training process.

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

As we konw, in the data the distribution of each label has huge difference. I decided to generate additional data for the labels, whose sizes are not at the average level(810), in oder that for these classes there are enough training set, which will also benifit to a better behavior of training later. After that, 
* The size of training set is 37392.
* The size of the validation set is 9348.
* The size of the test set stayed unchanged.
To add more data to the the data set, I used the following techniques for data augment: random translating, random scaling and random brightness. Which can also happen to the camera, when it works in different situations.

The difference between the original data set and the augmented data set is, that after data augmenting all data of classes seem more balanced and it makes sure that the training model will get a relative similar good behavior on each class. 
![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray-value image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				|
| Fully connected		| inputs 400 outputs 260 									|
| RELU					|												|
| Fully connected		| inputs 260 outputs 120 									|
| RELU					|												|
| Fully connected		| inputs 120 outputs 84 									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| inputs 84 outputs 43									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 
To avoid the underfitting, i developed LeNet deeper with adding more layers. 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
After reading the article from http://ruder.io/optimizing-gradient-descent/index.html#adam, https://www.dataquest.io/blog/learning-curves-machine-learning/, i got a better understand on parameter tuning of neural network.
With comparing the optimzer 'Adam', 'RMSprop' and 'Adadelta', finally i chose 'Adam', because Adam has a sligthly better performance.(reccording to the first link)
To tune the other parameters, I test different parameters combination. According to 2. link to compare the validation accuray and test accuracy to get the model out of low bias and high variance(because at the beginning the validation accuracy was already high and the test accuracy was relativ low). Finally using parameters EPOCHS = 20, BATCH_SIZE = 64(should match the memory architecture, usually being 32, 64, 128...). And according to the article, i used 0.001 as learning rate, and default values of other parameters of Adam. Another way to get out of low bias and hight variance, i used L2 regularization with the coefficient 0.01. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.984 
* test set accuracy of 0.931

Process: The LeNet was the basement of my model architecture. But the first prediction without any methods and data augment turns out to be failed with the validation accuracy 0.05(huge underfitting). When i generate the data(much better with accuracy about 0.8), and developed the model, the validation accuracy became much better and was about 0.9 finally but test accuracy was very low.  I tried different optimizers, tuned all the parameters and compare them to avoid overitting. After all steps the architecture has a good performance on both validation accuracy and test set accuracy.

Methods: for underfitting, generate data, develop the algorithm of model, decrease regularization. for overfitting, adding more traning set, increase the regularization.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

1. Priority road. This sign image is in low brightness(dark situation), small contrast to background, relative high jitteriness and taken in a upward angle. The background is a litte brighter but nearly the same brightness with sign. In background there is also a house, which could confuse the network. 
2. Yield. This sign image is in high brightness, big contrast to background, low jitteriness and taken in a horizontal angle. The background has much variation of brightness(bright and dark points).
3. No entry. This sign is in high brightness, big contrast to background, low jitteriness and taken in a upward and slightly rotated angle. The background has itself big contrast(blue sky and trees) and big variation.
4. No passing. This sign is in low brightness, relative small contrast to background, low jitteriness, and taken in a upward angle. The background has itself big contrast(gray sky and trees) and big variation.
5. speed limit (60 km/h). This sign is in high brightness, relative big contrast to background, low jitterness and taken in a upward angle. The background is very bright and big difference with the sign.
6. Stop. his sign is in high brightness, big contrast to background(red and blue), low jitterness and taken in a horizontal angle. The background is very bright and big difference with the sign. The brightness of background is also changing with the height of the image. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield   									| 
| Stop     			| Stop 										|
| Priority road					| Priority road											|
| No entry	      		| No entry					 				|
| speed limit(60 km/h)			| speed limit(20 km/h)					 				|
| No passing			| No passing      							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. Comparing to test set accuracy this is reasonable because the number of candidate images for prediction are too small. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Yield   									| 
| 1.00     				| Stop 										|
| 1.00     				| Priority road 										|
| 1.00					| No entry											|
| .56	      			| roundabout mandatory					 				|
| .99				    | No passing							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


