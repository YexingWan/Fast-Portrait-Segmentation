// testPreprocess.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#define _CRT_SECURE_NO_WARNINGS 
#include <opencv2/opencv.hpp>
#include <process.h>
#include <iostream>

using namespace cv;

int main()
{
	Mat img = imread("./test.jpg");
	tuple<int,int,bool> dresize_out = dynamic_resize(img, 192, 256, img);
	pad_32(img, 192, 256, img);
	imwrite("re_test_preprocess.jpg", img);
	return 0;
}

