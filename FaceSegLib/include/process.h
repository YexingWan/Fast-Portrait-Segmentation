#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

#if defined (_WIN32)
  #if defined(MyLibrary_EXPORTS)
    #define  MYLIB_EXPORT __declspec(dllexport)
  #else
    #define  MYLIB_EXPORT __declspec(dllimport)
  #endif /* MyLibrary_EXPORTS */
#else /* defined (_WIN32) */
 #define MYLIB_EXPORT
#endif

void MYLIB_EXPORT pad(cv::Mat &image, int expect_h, int expect_w, cv::Mat &dst);

tuple<int, int, bool> MYLIB_EXPORT dynamic_resize(cv::Mat &input_array, int expaect_h, int expect_w, cv::Mat &dst);

class MYLIB_EXPORT MGamma
{
public:
    MGamma();
    MGamma(float);
    void tran(cv::Mat &, cv::Mat &);

private:
    float fGamma;
    unsigned char lut[256];
};