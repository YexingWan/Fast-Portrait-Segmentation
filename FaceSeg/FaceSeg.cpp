// FaceSeg.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "process.h"
#include "model.h"

using namespace std;
using namespace cv;

double clockToMilliseconds(clock_t ticks)
{
    // units/(units/time) => time (seconds) * 1000 = milliseconds
    return (ticks / (double)CLOCKS_PER_SEC) * 1000.0;
}

int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--help") == 0)
    {
        cout << "./FaceSeg.exe $thread $blur_r $private_level" << endl;
        cout << "thread: [1-4] the number of thread used when do infer" << endl;
        cout << "blur_r: [2i+1] radius of blur kernel, 7-15 will get proper result" << endl;
        cout << "private_level: [0-4]: level of privacy" << endl;

        return 0;
    }

    auto model_path = ".\\Dnc_SINet_bi_256_192_fp16.mnn";
    auto bg_path = ".\\bg.jpg";
    auto face_weight = ".\\opencv_face_detector_uint8.pb";
    auto face_config = ".\\opencv_face_detector.pbtxt";
    float p[5] = {0.0, 0.41, 0.847, 1.38, 2.19};
    int skip_frequnece = 5;
    float gamma_tran = 0.6;
    bool no_skip = true;

    int thread = atoi(argv[1]);
    int blur_r = atoi(argv[2]);
    float private_level = p[atoi(argv[3])];

    cout << "thread(1-4):" << thread << endl;
    cout << "blur_r(2i+1):" << blur_r << endl;
    cout << "private_level(0-4):" << atoi(argv[3]) << endl;
    cout << "p:" << private_level << endl;

    cout << "model_path:" << model_path << endl;

    VideoCapture cap = VideoCapture(0);

    if (!cap.isOpened())

    {
        std::cout << "video not open." << std::endl;
        return 1;
    }

    // get info of video stream
    int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double rate = cap.get(CAP_PROP_FPS);
    bool stop(false);
    int have_face_detect_frequence = static_cast<int>(rate) * 1;
    int no_face_detect_frequence = static_cast<int>(rate * 0.3);

    cout << "have_face_detect_frequence：" << have_face_detect_frequence << endl;
    cout << "no_face_detect_frequence：" << no_face_detect_frequence << endl;

    // read model
    auto mnn_model = model(model_path, face_weight, face_config, bg_path, h, w, thread, private_level);

    // initial global val

    Mat frame;
    int idx = 1;
    tuple<int, int, bool> pre_out;
    cv::Mat infer_out;
    tuple<cv::Mat, cv::Mat> out;
    cv::Mat seg_frame;
    MGamma G = MGamma(gamma_tran);
    bool now_have_face = true;

    while (!stop)
    {

        if (!cap.read(frame))
        {
            std::cout << "no video frame" << std::endl;
            break;
        }
        // 此处为添加对视频的每一帧的操作方法
        //auto out = mnn_model.face_seg(frame);

        if (now_have_face && idx % have_face_detect_frequence == 0)
        {
            now_have_face = mnn_model.have_face(frame);
        }
        if (now_have_face)
        {
            if (no_skip || idx % skip_frequnece != 0)
            {
                cv::Mat ori_image;
                frame.copyTo(ori_image);
                G.tran(frame, frame);
                pre_out = mnn_model.preprocess(frame);
                infer_out = mnn_model.infer(frame);
                out = mnn_model.postprocess(infer_out, ori_image, get<0>(pre_out), get<1>(pre_out), get<2>(pre_out), true, blur_r);
                seg_frame = get<1>(out);
            }
            else
            {
                seg_frame = mnn_model.seg_by_mask(frame, get<0>(out));
            }
        }
        else
        {
            if (idx % no_face_detect_frequence == 0)
            {
                now_have_face = mnn_model.have_face(frame);
            }
            seg_frame = mnn_model.getBG();
        }
        idx += 1;
        cv::imshow("video", seg_frame);
        if (cv::waitKey(static_cast<int>(24 / thread)) > 0)
        {
            stop = true;
        }
    }
    cap.release();
    cout << "quit" << endl;
    return 0;
}
