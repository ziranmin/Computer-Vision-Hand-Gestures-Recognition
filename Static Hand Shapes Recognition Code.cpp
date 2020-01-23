
//
//  main.cpp
//  CS440 P2 OpenCV
//
//  Created by Ziran Min on 3/31/18.
//  Copyright © 2018 Ziran Min. All rights reserved.
//


/*    CS440 PS2 Computer Vision Hand Gesture Recognition
 *    Artificial Intelligence Spring 2018
 *
 *    --------------
 *    This program introduces the following concepts:
 *        a) Reading a stream of images from a webcamera, and displaying the video
 *        b) Skin color detection
 *        c) Background differencing
 *        d) Visualizing motion history
 *        e) Finding the contour, convex hull and defect of hand fingers
 *    --------------
 */

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

//function declarations

/**
 Function that returns the maximum of 3 integers
 @param a first integer
 @param b second integer
 @param c third integer
 */
int myMax(int a, int b, int c);

/**
 Function that returns the minimum of 3 integers
 @param a first integer
 @param b second integer
 @param c third integer
 */
int myMin(int a, int b, int c);

/**
 Function that detects whether a pixel belongs to the skin based on RGB values
 @param src The source color image
 @param dst The destination grayscale image where skin pixels are colored white and the rest are colored black
 */
void mySkinDetect(Mat& src, Mat& dst);

/**
 Function that does frame differencing between the current frame and the previous frame
 @param curr The current color image
 @param prev The previous color image
 @param dst The destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
 and previous image are not the same
 */
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst);

/**
 Function that accumulates the frame differences for a certain number of pairs of frames
 @param mh Vector of frame difference images
 @param dst The destination grayscale image to store the accumulation of the frame difference images
 */
void myMotionEnergy(vector<Mat> mh, Mat& dst);


int main()
{
    
    //----------------
    //Reading a stream of images from a webcamera, and displaying the video
    //----------------
    // For more information on reading and writing video: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
    // open the video camera no. 0
    VideoCapture cap(0);
    
    // if not successful, exit program
    if (!cap.isOpened())
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
    
    //create a window called "MyVideoFrame0"
    namedWindow("MyVideo0", WINDOW_AUTOSIZE);
    Mat frame0;
    
    // read a new frame from video
    bool bSuccess0 = cap.read(frame0);
    
    //if not successful, break loop
    if (!bSuccess0)
    {
        cout << "Cannot read a frame from video stream" << endl;
    }
    
    //show the frame in "MyVideo" window
    imshow("MyVideo0", frame0);
    
    //create a window called "MyVideo"
    // comment out for part 2
    //namedWindow("MyVideo", WINDOW_AUTOSIZE);
    //namedWindow("MyVideoMH", WINDOW_AUTOSIZE);
    namedWindow("Skin", WINDOW_AUTOSIZE);
    namedWindow("Contour", WINDOW_AUTOSIZE);
    namedWindow("Gesture", WINDOW_AUTOSIZE);
    
    /*
    // for part 2
    vector<Mat> myMotionHistory;
    Mat fMH1, fMH2, fMH3;
    fMH1 = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
    fMH2 = fMH1.clone();
    fMH3 = fMH1.clone();
    myMotionHistory.push_back(fMH1);
    myMotionHistory.push_back(fMH2);
    myMotionHistory.push_back(fMH3);
    */
     
    while (1)
    {
        // read a new frame from video
        Mat frame;
        bool bSuccess = cap.read(frame);
        imshow("MyVideo0", frame);
        //if not successful, break loop
        if (!bSuccess)
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        
        ///////////////////////////////////////
        // Part One: Static Gesture Recognition
        ///////////////////////////////////////
        
        // destination frame
        Mat frameDest;
        frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1); //Returns a zero array of same size as src mat, and of type CV_8UC1
        
        //----------------
        // Skin color detection
        //----------------
        
        mySkinDetect(frame, frameDest);
        imshow("Skin", frameDest);
        
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        
        // Find contours
        // For more information on contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
        findContours(frameDest, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        cout << "The number of contours detected is: " << contours.size() << endl;
        
        // Find detedcted convex hulls and defects of contours
        vector<vector<int> > hullsI(contours.size());
        vector<vector<Point> > hullsP(contours.size());
        vector<vector<Vec4i> > defects(contours.size());
        
        for (int i = 0; i <contours.size(); ++i) {
            convexHull(Mat(contours[i]), hullsI[i], false);
            convexHull(Mat(contours[i]), hullsP[i], false);
            convexityDefects(contours[i], hullsI[i], defects[i]);
        }
        
        Mat contour_output = Mat::zeros(frameDest.size(), CV_8UC3);
        
        // Find the contour that has the largest area and that is hand
        int max_area = 0;
        int max_index = 0;
        Rect bound_rec;
        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > max_area) {
                max_area = (int)area;
                max_index = i;
                bound_rec = boundingRect(contours[i]);
            }
        }
        
        Mat frame1 = frame;
        
        // Visualization of hand contours
        drawContours(contour_output, contours, max_index, Scalar(0, 0, 255), CV_FILLED, 8, hierarchy);
        drawContours(contour_output, contours, max_index, Scalar(255, 0, 0), 2, 8, hierarchy);
        drawContours(frame1, contours, max_index, Scalar(225, 0, 0), 2, 8, hierarchy);
        rectangle(contour_output, bound_rec, Scalar(0, 255, 0), 2, 8, 0);
        rectangle(frame1, bound_rec, Scalar(0, 0, 0), 1, 8, 0);
        
        imshow("Contour", contour_output);
        
        if (contours.size() > 0)
        {
            drawContours(frame1, contours, max_index, Scalar(0, 0, 255), 2, 8, hierarchy);
            drawContours(frame1, hullsP, max_index, Scalar(0, 255, 0), 2, 8, hierarchy);
        }
        
        
        // Find the center fo hand
        int countFingers = 0;
        Point2f handCenter;
        float radius;
        vector<Vec4i> maxDefects = defects[max_index];
        vector<Point> maxContour = contours[max_index];
        minEnclosingCircle(maxContour, handCenter, radius);
        circle(frame1, handCenter, 10, Scalar(0, 0, 255), 2, 8);
        
        // Find and count fingers
        for (int i = 0; i < maxDefects.size(); i++){
            int startIndx = maxDefects[i].val[0];
            Point fingerTip(maxContour[startIndx]);
            int depth = (int) (maxDefects[i].val[3]) / 256;
            
            if (depth > 15 && fingerTip.y < handCenter.y)
            {
                int start_idx = maxDefects[i].val[0];
                Point start = contours[max_index][start_idx];
                int end_idx = maxDefects[i].val[1];
                Point end = contours[max_index][end_idx];
                int far_idx = maxDefects[i].val[2];
                Point far = contours[max_index][far_idx];
                
                // connect detected hand's convex hulls and defects
                line(frame1, start, far, Scalar(255, 255, 0), 2);
                line(frame1, end, far, Scalar(255, 255, 0), 2);
                circle(frame1, start, 4, Scalar(124, 255, 255), 2);
                countFingers++;
            }
        }
        
        // recognize three hand shapes
        if (countFingers == 2){
            putText(frame1, "V Yeah!", Point(50, 100), FONT_HERSHEY_PLAIN, 4, Scalar(225, 0, 0), 3, 8);
        }
        
        if (countFingers == 5){
            putText(frame1, "High Five!", Point(50, 100), FONT_HERSHEY_PLAIN, 4, Scalar(225, 0, 0), 3, 8);
        }
        
        if (countFingers == 0 && contours.size() > 0){
            putText(frame1, "A Fist!", Point(50, 100), FONT_HERSHEY_PLAIN, 4, Scalar(225, 0, 0), 3, 8);
        }
        
        imshow("fingerVideo", frame1);
        
        //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        if (waitKey(30) == 27)
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
        
    }
    cap.release();
    return 0;
}


//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
    int m = a;
    (void)((m < b) && (m = b));
    (void)((m < c) && (m = c));
    return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
    int m = a;
    (void)((m > b) && (m = b));
    (void)((m > c) && (m = c));
    return m;
}

//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
    //Surveys of skin color modeling and detection techniques:
    //Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    //Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            //For each pixel, compute the average intensity of the 3 color channels
            Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
            int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
            if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B)){
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
}

// The following two are useful for my second part of code

//Function that does frame differencing between the current frame and the previous frame
void myFrameDifferencing(Mat& prev, Mat& curr, Mat& dst) {
    //For more information on operation with arrays: http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
    absdiff(prev, curr, dst);
    Mat cp = dst.clone();
    cvtColor(dst, cp, CV_BGR2GRAY);
    dst = cp > 50;
}

//Function that accumulates the frame differences for a certain number of pairs of frames
void myMotionEnergy(vector<Mat> mh, Mat& dst) {
    Mat mh0 = mh[0];
    Mat mh1 = mh[1];
    Mat mh2 = mh[2];
    
    for (int i = 0; i < dst.rows; i++){
        for (int j = 0; j < dst.cols; j++){
            if (mh0.at<uchar>(i, j) == 255 || mh1.at<uchar>(i, j) == 255 || mh2.at<uchar>(i, j) == 255){
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
}

