#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

void detect_keypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detector_type, bool bVis = false);
static void detect_keypoints_Harris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
static void detect_keypoints_ShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
static void detect_keypoints_Fast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
static void detect_keypoints_Brisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
static void detect_keypoints_Orb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
static void detect_keypoints_Akaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
static void detect_keypoints_Sift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);

void keypoints_descriptor(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);


void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

#endif /* matching2D_hpp */
