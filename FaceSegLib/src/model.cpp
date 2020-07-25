
#include <iostream>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <fstream>
#include "model.h"
#include "process.h"

using namespace MNN;
using namespace std;

float MEAN[3] = {(float)107.05166, (float)115.52001, (float)132.23209};
float STD[3] = {(float)64.02013, (float)65.14492, (float)68.3079};

model::model()
{
}

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
/// <returns></returns>
model::model(const char *weight_path, const char *face_w, const char *face_c,
			 const char *bg_path, int proper_bg_h, int proper_bg_w, int thread, float private_level) : weight_path{weight_path}, bg_path{bg_path}
{
	face_net = cv::dnn::readNetFromTensorflow(std::string(face_w), std::string(face_c));
	if (face_net.empty())
	{
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "prototxt:   " << face_w << endl;
		cerr << "caffemodel: " << face_c << endl;
		cerr << "Models are available here:" << endl;
		cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;
		cerr << "or here:" << endl;
		cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;
		exit(-1);
	}

	I = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(weight_path));
	MNN::ScheduleConfig config;
	auto type = MNN_FORWARD_CPU;
	config.type = type;
	config.numThread = thread;
	// If type not fount, let it failed
	config.backupType = type;
	BackendConfig backendConfig;
	// backendConfig.power = BackendConfig::Power_High;
	backendConfig.precision = BackendConfig::Precision_Low;
	//backendConfig.memory = BackendConfig::Memory_High;
	config.backendConfig = &backendConfig;
	MNN_PRINT("Start create Session!\n");

	S = I->createSession(config);
	if (nullptr == I)
	{
		std::cout << "Fail to creat Session!" << std::endl;
		exit(1);
	}

	MNN_PRINT("Create Session Success!\n");

	MNN_PRINT("Get input Tensor!\n");

	In_T = I->getSessionInput(S, NULL);

	MNN_PRINT("Get output Tensor!\n");
	privateLevel = private_level;
	Out_T = I->getSessionOutput(S, NULL);
	auto in_shape = In_T->shape();
	input_n = in_shape.at(0);
	input_c = in_shape.at(1);
	input_h = in_shape.at(2);
	input_w = in_shape.at(3);
	auto out_shape = Out_T->shape();
	output_n = out_shape.at(0);
	output_c = out_shape.at(1);
	output_h = out_shape.at(2);
	output_w = out_shape.at(3);
	auto syn_bg = cv::imread(bg_path);
	auto bg_h = syn_bg.rows;
	auto bg_w = syn_bg.cols;
	int re_bg_w = bg_w;
	int re_bg_h = bg_h;
	cout << "bg w:" << bg_w << endl;
	cout << "bg h:" << bg_h << endl;

	if (proper_bg_h and proper_bg_w)
	{
		cout << "proper_bg_w:" << proper_bg_w << endl;
		cout << "proper_bg_h:" << proper_bg_h << endl;
		if ((bg_h * 1.0) / (bg_w * 1.0) >= (proper_bg_h * 1.0) / (proper_bg_w * 1.0))
		{
			re_bg_w = (int)proper_bg_w + 4;
			re_bg_h = (int)(bg_h * ((proper_bg_w + (float)4.0) / bg_w));
		}
		else
		{
			re_bg_h = int(proper_bg_h + 2);
			re_bg_w = int(bg_w * ((proper_bg_h + (float)4.0) / bg_h));
		}
		cout << "resize bg w:" << re_bg_w << endl;
		cout << "resize bg h:" << re_bg_h << endl;
		cv::resize(syn_bg, syn_bg, cv::Size(re_bg_w, re_bg_h));
	}
	auto center_y = re_bg_h / 2;
	auto center_x = re_bg_w / 2;
	cv::Rect myROI(center_x - proper_bg_w / 2, center_y - proper_bg_h / 2, proper_bg_w, proper_bg_h);
	syn_bg(myROI).convertTo(bg, CV_32FC3);
}

int model::getInputH()
{
	return input_h;
}
int model::getInputW()
{
	return input_w;
}
int model::getInputC()
{
	return input_c;
}
int model::getInputN()
{
	return input_n;
}

int model::getOutputH()
{
	return output_h;
}
int model::getOutputW()
{
	return output_w;
}
int model::getOutputC()
{
	return output_c;
}
int model::getOutputN()
{
	return output_n;
}

Tensor *model::getOutputTensor()
{
	return Out_T;
}
std::shared_ptr<MNN::Interpreter> model::getInterpreter()
{
	return I;
}
Tensor *model::getInputTensor()
{
	return In_T;
}
Session *model::getSession()
{
	return S;
}
cv::Mat model::getBG()
{
	cv::Mat BG;
	bg.convertTo(BG, CV_8UC3);
	return BG;
}

int model::setBG(const char *new_bg_path, int proper_bg_h, int proper_bg_w)
{
	bg_path = new_bg_path;
	auto syn_bg = cv::imread(bg_path);
	auto bg_h = syn_bg.rows;
	auto bg_w = syn_bg.cols;
	int re_bg_w = bg_w;
	int re_bg_h = bg_h;
	if (proper_bg_h and proper_bg_w)
	{
		if (bg_h / bg_w >= proper_bg_h / proper_bg_w)
		{
			re_bg_w = (int)proper_bg_w + 2;
			re_bg_h = (int)(bg_h * ((proper_bg_w + (float)2.0) / bg_w));
		}
		else
		{
			re_bg_h = int(proper_bg_h + 2);
			re_bg_w = int(bg_w * ((proper_bg_h + (float)2.0) / bg_h));
		}
		cv::resize(syn_bg, syn_bg, cv::Size(re_bg_w, re_bg_h));
	}
	auto center_y = re_bg_h / 2;
	auto center_x = re_bg_w / 2;
	cv::Rect myROI(center_x - proper_bg_w / 2, center_y - proper_bg_h / 2, proper_bg_w, proper_bg_h);
	syn_bg(myROI).convertTo(bg, CV_32FC3);
	return 0;
}

int model::setPrivateLevel(float private_level)
{
	privateLevel = private_level;
	return 0;
}

tuple<int, int, bool> model::preprocess(cv::Mat &image)
{
	tuple<int, int, bool> pre_out = dynamic_resize(image, input_h, input_w, image);
	pad(image, input_h, input_w, image);
	cv::Mat input_imge;
	image.convertTo(image, CV_32FC3);
	subtract(image, cv::Scalar(MEAN[0], MEAN[1], MEAN[2]), image);
	vector<cv::Mat> channels(3);
	split(image, channels);
	channels[0] /= (STD[0] * 255.);
	channels[1] /= (STD[1] * 255.);
	channels[2] /= (STD[2] * 255.);
	merge(channels, image);
	return pre_out;
}

cv::Mat model::infer(cv::Mat image)
{
	auto nhwcTensor = MNN::Tensor::create<float>({1, input_h, input_w, 3}, image.data, MNN::Tensor::TENSORFLOW);
	In_T->copyFromHostTensor(nhwcTensor);
	I->runSession(S);
	Out_T->copyToHostTensor(nhwcTensor);
	auto result = nhwcTensor->host<float>();
	cv::Mat re = cv::Mat(cv::Size(output_w, output_h), CV_32FC2, result);
	return re;
}

tuple<cv::Mat, cv::Mat> model::postprocess(cv::Mat out, cv::Mat ori_image, int h_in, int w_in, bool re, bool seg_largest_only, int blur_r)
{
	// get mask with type CV_32F
	vector<cv::Mat> channels(2);
	split(out, channels);
	cv::Mat my_mask(output_h, output_w, CV_8U);
	cv::compare(channels[0] + privateLevel, channels[1], my_mask, cv::CMP_LT);
	auto roi = cv::Rect(0, 0, w_in, h_in);
	my_mask = my_mask(roi);
	auto cf = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(blur_r, blur_r));
	cv::morphologyEx(my_mask, my_mask, cv::MORPH_OPEN, cf);
	cv::morphologyEx(my_mask, my_mask, cv::MORPH_CLOSE, cf);

	if (seg_largest_only)
	{
		cv::Mat stats;
		cv::Mat centroids;
		int num = cv::connectedComponentsWithStats(my_mask, my_mask, stats, centroids);
		int max_area = -1;
		int max_idx = -1;
		if (num > 1)
		{
			for (int m = 1; m <= num - 1; m++)
			{
				if (stats.at<int>(m, cv::CC_STAT_AREA) > max_area)
				{
					max_area = stats.at<int>(m, cv::CC_STAT_AREA);
					max_idx = m;
				}
			}
			my_mask = my_mask == max_idx;
			my_mask.convertTo(my_mask, CV_8U);
			my_mask = my_mask * 255;
		}
	}
	int h = ori_image.rows;
	int w = ori_image.cols;

	if (re)
	{
		cv::resize(my_mask, my_mask, cv::Size(w, h), cv::INTER_NEAREST);
	}

	cv::GaussianBlur(my_mask, my_mask, cv::Size(blur_r, blur_r), 7.0);

	cv::Mat my_mask_32f;
	my_mask.convertTo(my_mask_32f, CV_32F);
	my_mask_32f /= 255.;

	// get seg img
	cv::Mat ori_image_f;
	ori_image.convertTo(ori_image_f, CV_32FC3);
	vector<cv::Mat> im_channels(3);
	vector<cv::Mat> bg_channels(3);
	split(ori_image_f, im_channels);
	split(bg, bg_channels);
	im_channels[0] = im_channels[0].mul(my_mask_32f) + bg_channels[0].mul(1.0 - my_mask_32f);
	im_channels[1] = im_channels[1].mul(my_mask_32f) + bg_channels[1].mul(1.0 - my_mask_32f);
	im_channels[2] = im_channels[2].mul(my_mask_32f) + bg_channels[2].mul(1.0 - my_mask_32f);
	cv::merge(im_channels, ori_image_f);
	cv::Mat result;
	ori_image_f.convertTo(result, CV_8UC3);
	return tuple<cv::Mat, cv::Mat>(my_mask_32f, result);
}

cv::Mat model::face_seg(cv::Mat image)
{

	cv::Mat ori_image;
	image.copyTo(ori_image);
	auto pre_out = preprocess(image);
	auto infer_out = infer(image);
	auto out = postprocess(infer_out, ori_image, get<0>(pre_out), get<1>(pre_out), get<2>(pre_out), true, 13);
	return get<1>(out);
}

cv::Mat model::seg_by_mask(cv::Mat ori_image, cv::Mat mask)
{
	// get seg img
	cv::Mat ori_image_f;
	ori_image.convertTo(ori_image_f, CV_32FC3);
	vector<cv::Mat> im_channels(3);
	vector<cv::Mat> bg_channels(3);
	split(ori_image_f, im_channels);
	split(bg, bg_channels);
	im_channels[0] = im_channels[0].mul(mask) + bg_channels[0].mul(1.0 - mask);
	im_channels[1] = im_channels[1].mul(mask) + bg_channels[1].mul(1.0 - mask);
	im_channels[2] = im_channels[2].mul(mask) + bg_channels[2].mul(1.0 - mask);
	cv::merge(im_channels, ori_image_f);
	cv::Mat result;
	ori_image_f.convertTo(result, CV_8UC3);
	return result;
}

bool model::have_face(cv::Mat image, float area_threashold)
{
	cv::Mat inputBlob = cv::dnn::blobFromImage(image, inScaleFactor,
											   cv::Size(face_inWidth, face_inHeight), cv::Scalar(104.0, 177.0, 123.0), false, false); //! [Prepare blob]

	face_net.setInput(inputBlob);			//! [Set input blob]
	cv::Mat detection = face_net.forward(); //! [Make forward pass]
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	for (int i = 0; i < detectionMat.rows; ++i)
	{
		// judge confidence
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			float area = (detectionMat.at<float>(i, 5) - detectionMat.at<float>(i, 3)) * (detectionMat.at<float>(i, 6) - detectionMat.at<float>(i, 4));
			if (area > area_threashold)
			{
				return true;
			}
		}
		return false;
	}
	return false;
}

model::~model()
{
}
