# SFND 3D Object Tracking

[//]: # (Image References)

[image1]: ./images/course_code_structure.png "Architecture"
[image2]: ./images/detection.gif "Detection"
[image3]: ./images/lidarttc.png "Lidar TTC"
[image4]: ./images/image_16.png "Img1"
[image5]: ./images/image_17.png "Img2"
[image6]: ./images/image_18.png "Img3"

![alt text][image2]
# Project overview
Bellow I will address each point in the [project rubric](https://review.udacity.com/#!/rubrics/2550/view)

The overall architecture of the project is described in this image:
![alt text][image1]

## FP.1 Match 3D Objects
The first step in implementing this project was the keypoint matching. Using a combination of detectors and descriptors to extract the object keypoints on a sequence of images we are able to match keypoints in between two consecutive images. These are keypoints matches. Also using YOLO deep learning algorithm we can get the bounding boxes on each object on the image.

The object of this step is to achieve matching of bounding boxes detected by the YOLO algorithm. This is done by applying the following steps:
- Looping over the keypoint matches for the previous and current frame and determining which keypoint belongs to which bounding box 
- Storing the bounding box ids in a multimap. A multimap is used because id allows multiple pairs that have the same key value
- Looping over all the bounding boxes in the current frame and counting the number of occurrences for each matches bounding box in previous image and counting all the matches 
- Determining the maximum number of occurrences for a match of bounding box pairs 
## FP.2 Compute Lidar-based TTC
Computing TTC based on lidar measurements is done by first filtering lidar points. We want to eliminate the points that are in the lanes different from ego lane and the points that are reflection of the vehicle hood. We only use the lidar points which have the y values that indicates that they are in the vehicle ego lane. For filtering the x values we take the average values of all the lidar points x values in the vehicle ego lane. Using this we take an average closest x values of lidar points. This is done for both the previous and current frame. After filtering lidar points and extracting the average closest x, then we only apply the TTC formula for the constant velocity model. 

![alt text][image3]
## FP.3 Associate Keypoint Correspondences with Bounding Boxes
For this step we loop over the keypoint matches for the previous and the current frame to determine which current frame keypoints are contained in the bounding box region of interest. Because there are outlies among the keypoints matches we filter the matches using a distance mean threshold. We determine the mean distance between keypoints mathces and scale it by 0.75 and take only the matches that are bellow that threshold. 
## FP.4 Compute Camera-based TTC
After performing keypoint correspondences with bounding boxes we use the perform the TTC computation for the camera data.
## FP.5 Performance Evaluation 1
The main errors in the Lidar-based TTC computation come from the Lidar characteristics. Lidar points are sometimes reflected by the hood of the ego vehicle. These points make it appear as if there is an obstacle really close and they need to be filtered out. Other issues come from unstable lidar points reflected by the preceding vehicle reflective surfaces. 

In the last three images in the tested sequence we se a jump in the TTC calculation for the Lidar. 

![alt text][image4]

![alt text][image5]

![alt text][image6]

## FP.6 Performance Evaluation 2
After running all the detector-descriptor to compare the performance of each combination we see that detectors such as HARRIS and ORB produce unreliable results. HARRIS performed the lowest due to is poor keypoints detections. Detectors such as FAST, SIFT and AKAZE produce the most stabile results with no oscillations of the TTC measurements in between frame. Others combinations produce some jumps in the TTC calculation in between frames. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.
