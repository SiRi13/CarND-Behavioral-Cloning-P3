# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia]:       ./images/nvidia_cnn.png "NVIDIA End-To-End Learning Model"
[image1]:       ./images/cArI_visualized.png "Model Visualization"
[valLossPlot]:  ./images/plots/val_loss_plot_nvidia_with_pooling_20170411_181923.jpeg "Validation versus Training loss of Model"
[image2]:       ./simulator_data/tr1_lap/IMG/center_2017_04_10_22_51_55_322.jpg "Regular Image"
[image3]:       ./simulator_data/tr1_turn3/IMG/right_2017_04_11_20_17_53_952.jpg "Recovery Image"
[image4]:       ./simulator_data/tr1_turn3/IMG/right_2017_04_11_20_17_54_435.jpg "Recovery Image"
[image5]:       ./simulator_data/tr1_turn3/IMG/right_2017_04_11_20_17_53_952.jpg "Recovery Image"
[image6]:       ./images/random_batch/batch_image234.jpeg "Normal Image"
[image7]:       ./images/plots/random_batch.png "Flipped Image"
[image8]:       ./images/plots/random_augmented_images.png "Preprocessed Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* cArI-network.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Udacity provided the simulator and the **drive.py** file, which connects to the simulator, retrieves a frame, feeds it into the network and takes its prediction to update the steering angle.
I did not have to change the **drive.py** script because my normalization is done by the model with keras _**Cropping2D**_ and _**Lambda**_ layers.

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The **model.py** file contains the code for training and saving the convolutional neural network.

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

I started with an similar implementation as the original NVIDIA network, which did not yield any positive results. The car usually made the first turn and crossed the bridge but had always problems with the second turn, which does not have the same border as the rest of the track.
After this setback I tried using the same network architecture as I used in my Traffic Sign Project.
The validation and training losses looked better and so was the driving of the car.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]
![alt text][image8]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
