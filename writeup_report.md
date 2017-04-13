# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]:       ./images/model_plot.png "Model Visualization"
[fm1_with_street]:       ./images/plots/layer1_feature_map_with_street.png "Model Visualization"
[fm1_without_street]:    ./images/plots/layer1_feature_map_withou_street.png "Model Visualization"
[fm2_with_street]:       ./images/plots/layer2_feature_map_with_street.png "Model Visualization"
[fm2_without_street]:    ./images/plots/layer2_feature_map_withou_street.png "Model Visualization"
[valLossPlot]:  ./images/plots/val_loss_plot_nvidia_with_pooling_20170411_181923.jpeg "Validation versus Training loss of Model"
[image2]:       ./simulator_data/tr1_lap/IMG/center_2017_04_10_22_51_55_322.jpg "Regular Image"
[image3]:       ./simulator_data/tr1_turn3/IMG/right_2017_04_11_20_17_53_952.jpg "First Recovery Image"
[image4]:       ./simulator_data/tr1_turn3/IMG/right_2017_04_11_20_17_54_435.jpg "Middle Recovery Image"
[image5]:       ./simulator_data/tr1_turn3/IMG/right_2017_04_11_20_17_53_952.jpg "Last Recovery Image"
[image6]:       ./images/random_batch/batch_image234.jpeg "Normal Image"
[image7]:       ./images/plots/random_batch.png "Flipped Image"
[image8]:       ./images/plots/random_augmented_images.png "Preprocessed Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* run1.mp4 video of one complete lap in autonomous mode

#### 2. Submission includes functional code
Udacity provided the simulator and the **drive.py** file, which connects to the simulator, retrieves a frame, feeds it into the network and takes its prediction to update the steering angle.
I did not have to change the **drive.py** script because my normalization is done by the model with _keras_ _**Cropping2D**_ and _**Lambda**_ layers.

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The **model.py** file contains the code for training and saving the convolutional neural network.
Loading, preprocessing and augmenting the data points is handled by the file **data_generator.py**.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a three layer convolutional neural network with layer depths of 16, 32 and 64 and a 3x3 kernel as well as an *relu* activation. Each is followed by a pooling layer with standard 2x2 pooling (**model.py** lines 39-44).  
After flatten the last pooling layer in line 45 I added three fully connected layer with 500, 100 and 20 units with *relu* activation respectively. Dense layer one and two are followed by 50% dropouts whereas after dense layer three the output dense layer follows right away (**model.py** lines 46-52).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (**model.py** lines 47 and 49).

The model was trained and validated on different data sets to ensure that the model was not overfitting (**model.py** line 14). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an _Adam_ optimizer, so the learning rate was not tuned manually (**model.py** line 54).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
At first I only used one lap of driving as intended and one lap of driving in the opposite direction.
To improve the driving through turns and over the bridge I added two additional sets of data for those specific parts of the first track.
In the end I trained on two full laps of each direction, two sets of bridge data and two sets of turns data. Adding the Udacity provided data set did not improve the result and I excluded it.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with an similar implementation as the original NVIDIA network, which did not yield any positive results.
The car usually made the first turn and crossed the bridge but had always problems with the second turn, which does not have the same border as the rest of the track.
After this setback I tried using the same network architecture as I used in my Traffic Sign Project.
The validation and training losses looked better and so was the driving of the car.

To improve the training-testing-workflow of the model I implemented the _**EarlyStopping**_ and _**ModelCheckpoint**_ callbacks of the _keras_ framework.  
I set the early stopping to halt after four epochs without improvement which I first had set to 0.001 but lowered it to 0.008 which provided the best results.  
The _**ModelCheckpoint**_ callback saved the weights automatically after each epoch and with the *save_best_only* flag set to _True_, I always ended up having the best result, even if the model trained too long.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 35-52) consisted of a convolution neural network with the following layers and layer sizes:

```
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 77, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 75, 318, 16)       448
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 37, 159, 16)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 35, 157, 32)       4640
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 17, 78, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 76, 64)        18496
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 7, 38, 64)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 17024)             0
_________________________________________________________________
dense_1 (Dense)              (None, 500)               8512500
_________________________________________________________________
dropout_1 (Dropout)          (None, 500)               0
_________________________________________________________________
dense_2 (Dense)              (None, 100)               50100
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_3 (Dense)              (None, 20)                2020
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 21
=================================================================
Total params: 8,588,225.0
Trainable params: 8,588,225.0
Non-trainable params: 0.0
_________________________________________________________________
```

Here are two feature maps of the first convolutional layer.
The source image of the first map contained a street whereas the second did not.
It is visible that the network can detect streets or border of the street.

![alt text][fm1_with_street]
Feature Map with street in source image
![alt text][fm1_without_street]
Feature Map without street in source image  

The following diagram shows the training progress of the network and that it stopped after 8 epochs with a validation loss of below 0.12.

![alt text][valLossPlot]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.
Here is an example image of center lane driving:

![alt text][image2]

I did not create recovering data because it is very difficult to create it and made the result worse than without.

To increase the number of images without having to drive the simulator all day, I augmented each data point randomly.
Each data point was added as it was. Then it either got a shadow overlay or the brightness got changed.
These two alterations are exclusive, meaning there was only applied one of them.
After that I removed between none and ten lines of the top and the bottom and flipped the image and the angle respectively.
None of these four methods were applied every time but with a fifty-fifty chance.

![alt text][image8]


There were 5214 data points loaded. After preprocessing and augmentation there are probably about 11260 points to feed into the network.
The data gets shuffled before splitting it into training and validation data as well as before creating the batches and
before yielding them from the generator.
For training I used 4171 of the 5214 data points or 80% whereas the left 1042 points were used for validation.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 8 which was determined by the _**EarlyStopping**_ callback.
As optimizer I used _Adam_ with a start learning rate of 0.0001. As batch size I set 32.
The loss function was Mean Average Error to prevent the model from always predicting 0.0.
