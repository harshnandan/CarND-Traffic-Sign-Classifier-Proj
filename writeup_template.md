# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./writeup_images/trainingDataSet_stats.png "Training dataset stats"
[image2]: ./writeup_images/trainingDataSet.png "Training dataset"
[image3]: ./writeup_images/brightness_contrast.png "Brighness and Contrast"
[image4]: ./writeup_images/sharp.png.png "Sharpen"
[image5]: ./writeup_images/histogram_eq.png "Histogram Equalization"
[image6]: ./writeup_images/scale_crop.png "Brighness and Contrast"
[image7]: ./writeup_images/rotate.png "Brighness and Contrast"
[image8]: ./writeup_images/combined_operation.png "sample Augmented Image"

[image9]: ./writeup_images/new_images.png "New images"
[image10]: ./writeup_images/prediction.png "Prediction"
[image11]: ./writeup_images/softmax_prediction.png "Top 5 predictions"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/harshnandan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

It is easy to realize that the colors of traffic signs are important and hence converting images to gray scale may not be the best option. Limited testing with gray scale images were done, but it did not perform consistent performance improvement when compared to colored images.

The provided training data is augmented with images which are modified version of randomly sampled images within the training data set. This not only provides richer training data set, but also modifies the quality of several of the images. The applied transforms can be broadly divided into two categories:

1. Image quality transforms :

    a. Adjust Contrast and Brightness
	
![alt text][image3]

    b. Sharpen the image
	
![alt text][image4]

    c. Histogram Equalization
	
![alt text][image5]

2. Image shape transforms :

    a. Scale image and crop

![alt text][image6]

    b. Rotate

![alt text][image7]

One transformation from each of the above mentioed category is randomly picked an applied sequentially to randomly selected image. The user specifies the number of figures to augment to original training dataset.

Here is an example of an original image and an augmented image:

![alt text][image8]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   				|
| Fully Connected    	| Input 1600, outputs 120                    	|
| RELU					|												|
| Fully Connected    	| Input 120, outputs 84                      	|
| RELU					|												|
| Fully Connected    	| Input 84, outputs 10                      	|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The network was trained using AdamOptimizer (learning rate= 0.001) with a batch size of 64 and over 40 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97.99%
* validation set accuracy of 94.17%
* test set accuracy of 92.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The initial architecture had only 12 and 16 activation layers in the first two convolutional layers. 

* What were some problems with the initial architecture?

It was noticed that both training and cros-validation set accuracy plateued at around 96% and 90% respectively.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The plateu indicated that the model was underfitting and needs more parameters to lear from the training data. 

* Which parameters were tuned? How were they adjusted and why?

Hence the I started increasing the number of activation layers. The presented architecture provided me with the needed accuracy on a consistent basis.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Addition of dropuot of (keep probability = 0.8) further helped the cross-validation performance.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

6 new German traffic sign images were downloaded from the internet. The second image shown below is deliberately chosen to be a hard image. The image is taken in limited illumination and almost half of the sign is covered with snow. 

![alt text][image9] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                    |     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| Right of way on next Intersection | Right of way on next Intersection				| 
| Road Work             			| Road Work  									|
| Speed Limit (70 km/h)				| Speed Limit (70 km/h)							|
| General Caution          			| General Caution								|
| Wild animal crossing			    | Wild animal crossing 							|

![alt text][image10] 

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. But the trained model was again testing on 6 more images which were variant of the original 6 images in terms of brightness, contrast, histogram, scale and rotation. All the test images were labeled correctly.  

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all the 12 images (6 original and 6 variants) the model shows great confidence in predicted labels. Even the sign which was partially covered with snow is labeled correctly.

![alt text][image11] 

