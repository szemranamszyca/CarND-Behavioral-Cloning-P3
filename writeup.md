# **Behavioral Cloning** 

## Arkadiusz Konior - Project 3.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia]: ./imgs/nVidia_model.png "Nvidia model architecture"
[histogram]: ./imgs/histogram.png "Angles' histogram"
[example]: ./imgs/data_example.jpg "Image data example"
[cropped]: ./imgs/data_example_cropped.jpg "Image data example"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run_track1.mp4 showing successful run on track1
* run_track2.mp4 unsuccessful run on track2, but still surprisingly good - more about that further.

#### 2. Submission includes functional code
With the use of the simulator provided by Udacity and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The code is quite simple (it implements nVidia architecture presented during lessons), with extra function to plot histogram. Also, in drive.py at line 65, I've converted imgage from RGB (PIL) to BGR (default  for OpenCV - model was trained on that color space)

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. As I mentioned, it is nVidia network architecture with one extra dropout layer and one image cropping layer.

At first, I opened .csv file and using csv.reader I got all the information from it. As the next step, in simple "for loop", I loaded all images and measurements which I had used in order to train a Neural Network.

### Model Architecture and Training Strategy

#### 1. nVidia network architecture (presented during lessons)

I've decided to use suggested nVidia architecture with some extra layers:
* Lambda layer at the beginning to normalize values
* Cropping layer to cut-off irrelevant part of image
* Dropout layer after flatten layer to avoid overfitting (keep probability = .5)

![Model][nvidia]

* RELU layers introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18). 
* Data was split to train/valid sets (line 70)
* Adam optimizer was used 
* To train model, I've used only camera from the center of vehilice, without ANY augmentation.
* Data was shuffled
* Netowrk was trained for 8. epoches

#### 2. Collection strategy

I've read a lot of materials while preparing for this project, and they have got one thing in common: everyone said that this project is about DATA. Reading others reports, I've noticed that angle 0 is biased, which is correct - while your car is running straight ahead you don't have to do much. So I decided to take up a completly diffrent approach... 

I've tried to be the worst driver ever. I was constatly bouncing from one side to another, only "oscillating" around middle of the road. I took one lap. Let's take a look at my histogram:

![Angles' histogram][histogram]

And the example of training set (before cropping)

![Example image - center][example]

Cropped image (70px from top, 20px from bottom)

![Example cropped image - center][cropped]


Now, the extrem values of angle are biased. I've trained my network (only 1184 samples for train and 296 for validation!) to start autonomous driving and... it made it! It finished one lap. (file  - *run_track1.mp4*)

Even on the second track (no samples taken there at all) it was going surprisingly well. It didn't finished it (the car had accident on the bridge), but taking into account number of samples - it's still a good result. (file  - *run_track2.mp4*)

### Discussion

I think that the major reason why it works is quite low speed at autonomus mode ( 9 ). During testing at higher speeds, the car was "oscillating" and fell out of the way.

As I could see, the steering data were sending to autonomous model with high freqency. At speed 9, while the car was getting closer to the edge of the road, trajectory was almost immediately corrected (biased extrem values of angles). 

With the bug fixed (covert image from RGB to BGR) at higher speed ( 15 ) oscillation is still visible, but the vehilice finished one lap without any other problems *speedup.mp4*.
