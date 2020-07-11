#pragma once
#ifndef DLL_EXPORT
#define DLL_EXPORT __declspec(dllexport)
#endif


#include <MNN/Tensor.hpp>
#include <MNN/Interpreter.hpp>
#include <opencv.hpp>


using namespace MNN;
using namespace std;


class DLL_EXPORT model {
public:
	model();

	/// <summary>
	/// 主要的构造函数，初始化必要的模型类
	/// </summary>
	/// <param name="weight_path">MNN模型文件路径</param>
	/// <param name="face_w">用于人脸识别的权重文件路径</param>
	/// <param name="face_c">用于人脸识别的config文件路径</param>
	/// <param name="bg_path">背景路径</param>
	/// <param name="proper_bg_h">合适的背景高度，一般设置为视频流的高度</param>
	/// <param name="proper_bg_w">合适的背景宽度，一般设置为视频流的宽度</param>
	/// <param name="thread">推理用线程数</param>
	/// <param name="private_level">隐私参数（概率转换相关），隐私程度从低到高建议数值为【0.0,0.41,0.847,1.38,2.19】，</param>
	/// <returns></returns>
	model(const char* weight_path, const char* face_w, const char* face_c, 
		const char* bg_path, int proper_bg_h = NULL, int proper_bg_w = NULL, 
		int thread = 1, float private_level = 0.41);
	~model();
	int getInputH();
	int getInputW();
	int getInputC();
	int getInputN();
	int getOutputH();
	int getOutputW();
	int getOutputC();
	int getOutputN();
	cv::Mat getBG();
	int setBG(const char* new_bg_path, int proper_bg_h = NULL, int proper_bg_w = NULL);
	int setPrivateLevel(float private_level);

	/// <summary>
	/// 图像预处理模块，直接对输入Mat做操作，输出预处理得到的三个参数，分别为postprocess方法需要用到的h_in, w_in, 和 re
	/// </summary>
	/// <param name="image">输入图像</param>
	/// <returns></returns>
	tuple<int, int, bool> preprocess(cv::Mat& image);
	/// <summary>
	/// 推理，将预处理后的图像输入该方法，得到经过网络推理后的结果矩阵，类型为Mat
	/// </summary>
	/// <param name="frame">预处理后的图像</param>
	/// <returns></returns>
	cv::Mat infer(cv::Mat frame);
	/// <summary>
	/// 后处理，输入网络推理的结果，原图，前处理得到的三个参数与两个设置参数，输出两个Mat，分别为最终的Mask和经过bg覆盖背景后的图像
	/// </summary>
	/// <param name="out">infer方法返回的结果矩阵</param>
	/// <param name="ori_image">原图</param>
	/// <param name="h_in">preprocess输出的第一个参数</param>
	/// <param name="w_in">preprocess输出的第二个参数</param>
	/// <param name="re">preprocess输出的第三个参数</param>
	/// <param name="seg_largest_only">是否只分割最大的区域</param>
	/// <param name="blur_r">边缘轮廓平滑的半径（单数，7-15为佳）</param>
	/// <returns>Mask矩阵，经过bg覆盖背景后的图像</returns>
	tuple<cv::Mat, cv::Mat> postprocess(cv::Mat out, cv::Mat ori_image, int h_in, int w_in, bool re, bool seg_largest_only, int blur_r);

	/// <summary>
	///		一步式api，输入为原图像，输出为经过bg覆盖背景后的图像
	/// </summary>
	/// <param name="image"></param>
	/// <returns></returns>
	cv::Mat face_seg(cv::Mat image);

	/// <summary>
	///		输入postprocess输出的第一个矩阵Mask，以及一张原图，输出根据该Mask生成经过bg覆盖背景后的图像，该方法可以用于截帧预测（既2帧预测一次，2帧用同一个Mask）
	/// </summary>
	/// <param name="ori_image">原图</param>
	/// <param name="mask">postprocess输出的第一个矩阵Mask</param>
	/// <returns>经过bg覆盖背景后的图像</returns>
	cv::Mat seg_by_mask(cv::Mat ori_image, cv::Mat mask);

	/// <summary>
	///		人脸识别模块，输入原图和人脸区域占全图面积比阈值，返回是否存在面积大于该阈值的人脸
	/// </summary>
	/// <param name="image">原图</param>
	/// <param name="area_threashold">人脸区域占全图面积比阈值，比如人脸的面积占全图的0.25*0.25=0.0625，既约等于长宽为原图的1/4</param>
	/// <returns>bool，是否存在面积大于该阈值的人脸</returns>
	bool have_face(cv::Mat image,float area_threashold = 0.0625);


private:
	Tensor* getOutputTensor();
	std::shared_ptr<MNN::Interpreter> getInterpreter();
	Tensor* getInputTensor();
	Session* getSession();
	cv::Mat bg; // 背景
	const char* weight_path; // mnn模型路径
	const char* bg_path; //背景路径
	Session* S; //MNN Session
	MNN::Tensor* In_T; //Input Tensor
	MNN::Tensor* Out_T; //Output Tensor
	std::shared_ptr<MNN::Interpreter> I; // MNN Interpreter

	int input_w;
	int input_h;
	int input_c;
	int input_n;
	int output_w;
	int output_h;
	int output_c;
	int output_n;
	float privateLevel;

	cv::dnn::Net face_net; //dnn::Net for face detection
	const size_t face_inWidth = 300;
	const size_t face_inHeight = 300;
	const double inScaleFactor = 1.0;
	float confidenceThreshold = 0.5; // face confident threshold


};