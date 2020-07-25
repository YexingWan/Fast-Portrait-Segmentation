#pragma once
#ifndef DLL_EXPORT
#define DLL_EXPORT __declspec(dllexport)
#endif

#include <opencv.hpp>
#include <iostream>

using namespace std;

DLL_EXPORT void pad(cv::Mat& image, int expect_h, int expect_w, cv::Mat& dst);

DLL_EXPORT tuple<int, int, bool> dynamic_resize(cv::Mat& input_array, int expaect_h, int expect_w, cv::Mat& dst);



class DLL_EXPORT gamma {
public:
    gamma();
    gamma(float);
    void tran(cv::Mat&, cv::Mat& );


private:
    float fGamma;
    unsigned char lut[256];
};