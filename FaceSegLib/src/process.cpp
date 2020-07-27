#include "process.h"

/// <summary>
/// pad image to expect size with background of 127
/// </summary>
/// <param name="image"></param>
/// <param name="expaect_h"></param>
/// <param name="expect_w"></param>
/// <param name="dst"></param>
void pad(cv::Mat &image, int expect_h, int expect_w, cv::Mat &dst)
{

    int img_h = image.rows;
    int img_w = image.cols;
    if (img_h > expect_h || img_w > expect_w)
    {
        dynamic_resize(image, expect_h, expect_w, image);
        img_h = image.rows;
        img_w = image.cols;
    }
    cv::Mat background(cv::Size(expect_w, expect_h), CV_8UC3, cv::Scalar(127, 127, 127));
    cv::Rect ROI(0, 0, img_w, img_h);
    image.copyTo(background(ROI));
    dst = background;
}

/// <summary>
/// 等比例resize输入图片，使在expect_h * expect_w范围内能囊括该图片
/// </summary>
/// <param name="input_img">cv::Mat 类型，需要resize的图片</param>
/// <param name="expect_h">期望范围的h</param>
/// <param name="expect_w">期望范围的w</param>
/// <returns> tuple<int,int,re> 第一个是resize后的 </returns>
tuple<int, int, bool> dynamic_resize(cv::Mat &input_img, int expect_h, int expect_w, cv::Mat &dst)
{
    int w = input_img.cols;
    int h = input_img.rows;
    int h_in = h;
    int w_in = w;
    bool re = false;
    if (h > expect_h || w > expect_w)
    {
        re = true;

        if (h * 1.0 / w >= expect_h * 1.0 / expect_w)
        {
            h_in = expect_h;
            w_in = (int)(w * (expect_h * 1.0 / h));
        }
        else
        {
            w_in = expect_w;
            h_in = (int)(h * (expect_w * 1.0 / w));
        }
    }
    tuple<int, int, bool> result(h_in, w_in, re);
    cv::resize(input_img, dst, cv::Size(w_in, h_in));
    return result;
}

MGamma::MGamma()
{
    fGamma = 0.455;
    for (int i = 0; i < 256; i++)
    {
        lut[i] = static_cast<uchar>(pow((i / 255.0f), fGamma) * 255.0f);
    }
}

MGamma::MGamma(float fgamma)
{
    fGamma = fgamma;
    for (int i = 0; i < 256; i++)
    {
        lut[i] = static_cast<uchar>(pow((i / 255.0f), fGamma) * 255.0f);
    }
}

void MGamma::tran(cv::Mat &src, cv::Mat &dst)
{
    dst = src.clone();
    cv::MatIterator_<cv::Vec3b> it, end;
    for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
    {
        (*it)[0] = lut[((*it)[0])];
        (*it)[1] = lut[((*it)[1])];
        (*it)[2] = lut[((*it)[2])];
    }
}
