#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

void pad(cv::Mat &image, int expect_h, int expect_w, cv::Mat &dst);

tuple<int, int, bool> dynamic_resize(cv::Mat &input_array, int expaect_h, int expect_w, cv::Mat &dst);

class MGamma
{
public:
    MGamma();
    MGamma(float);
    void tran(cv::Mat &, cv::Mat &);

private:
    float fGamma;
    unsigned char lut[256];
};