# Fast Portrait Segmentation

Portrait Segmentation is a common front-end task for many application as proprocessing process. 
This project provide a simple portrait segmentation api for developer to "plugin and play".
This project is base on [MNN](https://github.com/alibaba/MNN) inference engine. 

This project is for the function of dynamic background in online-course application.

The solution model based on [SINET](https://arxiv.org/abs/1911.09099), which is a extreme light-weight model for portrait segmentation. 

Another solution is base on Depth Map Prediction from single image, which should get more stable result.
The solution is in developing and will come out soon.


# Requirement

This project is developed on VS2019 and only test on Windows. We now provide CMAKELIST to support multi-platform by cmake.

* CMAKE
* MNN 1.0.0 
* OpenCV >= 4.1
* VS2019 (For Windows)
* [Visual C++ Redistributable Packages](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) (for run the demo on Windows)

The project can run the realtime segmentation on relatively low-level cpu (i3-4300U with 4GB memory) with only around 30%-40% cpu usage.


# Demo

Python demo and C++ demo is given in the project. The python demo is in folder pyFaceSeg. C++ demo is FaceSeg.exe or FaceSeg on different system

## Python Demo

Python demo is the intermediate outcome from developing C++ demo. Start from pytorch model (origin SINet), the weight should
 be convert to onnx first, then use convert tools (mnnconvert.py) given by mnn to convert the model to mnn format.<br>
 
### File list

* bg.jpg: The background image
* cam_seg.py: The main python script of python demo
* XXX.mnn: model with mnn format
* XXX.onnx: model with onnx format (can be checked by [netron](https://github.com/lutzroeder/netron)) 
* opencv_face_*: opencv dnn face detection model, it is not used in python demo
* runIE: script for testing the correctness of model conversion
* test_opencv_facedet.py: script for testing opencv face detection

### Run python demo

* install requirement
```shell
pip install -r requirements
``` 
* run script
```shell
python cam_seg.py 
``` 


## C++ Demo

The minimal requirement file for runing demo is given in folder /x64/Run.
* bg.jpg: The background used in demo.
* Dnc_SINet_bi_256_192_fp16.mnn: MNN model for portrait segmentation
* MNN.dll: the MNN released dll (Please download yourself on MNN release site: https://github.com/alibaba/MNN/releases)
* opencv_face_*: model files for face detection in demo.

### Build demo on Windows

* Make sure the VS in install correctly 
* Generate FaceSeg.sln file by cmake
* Load the project by file FaceSeg.sln in VS
* Set the external link correctly to make sure VS can find MNN, OpenCV and other requirement.
* The project use **OpenCV static link library** as default. if you want to use dynamic version of OpenCV, 
make sure the project is setting correct in VS and copy all OpenCV dll to the same folder of executable file when run the demo.
* Build the FaceSegDll solution first.
* Then build the FaceSeg solution.

An executable file FaceSeg.exe and a dll file FaceSegDll.dll can be found in x64/Release/. 

### Build demo on Linux

* Please make sure OpenCV and MNN can be found by CMake

```shell
 export MNN_INCLUDE_DIRS=/path/to/MNN/inc
 export MNN_LIBRARY_DIRS=/path/to/MNN.so
 cd /path/to/project
 mkdir build
 cd build && cmake .. && make
 ```

### Run demo

For Windows, copy FaceSeg.exe and FaceSegDll.dll in x64/Run. Then open the powershell and run by script

```shell
cd path/to/project/Release/Run
./FaceSeg.exe --help
```
For Linux, copy FaceSeg.exe and FaceSegDll.dll in x64/Run. Then open Termial and run
```shell
cd path/to/project/Release/Run
./FaceSeg --help
```

The help information will show on control panel:

```log
./FaceSeg.exe $thread $blur_r $private_level
thread: [1-4] the number of thread used when do infer
blur_r: [2i+1] radius of blur kernel, 7-15 will get proper result
private_level: [0-4]: level of privacy
```

## Demo example

![demo](https://github.com/YexingWan/Fast-Portrait-Segmentation/blob/master/WechatIMG21214.png)


# DLL

The all-in-one dll file is FaceSegDll.dll. Read the head file model.h and process.h for detail.
