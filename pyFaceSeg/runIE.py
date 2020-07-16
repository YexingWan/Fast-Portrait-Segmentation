from argparse import ArgumentParser
import numpy as np
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def pad_32(img:np.ndarray,expect_h=None,expect_w = None):
    if expect_h is not None:
        assert (expect_h % 32 == 0)
    if expect_w is not None:
        assert (expect_w % 32 == 0)

    img_h,img_w,_ = img.shape
    h = ( img_h // 32 + 1 ) * 32 if img_h % 32 != 0 else img_h
    w = ( img_w // 32 + 1 ) * 32 if img_w % 32 != 0 else img_w
    expect_h = h if expect_h is None or expect_h < h else expect_h
    expect_w = w if expect_w is None or expect_w < w else expect_w
    bg = np.zeros(((int(expect_h),int(expect_w),3)))
    bg[:img_h,:img_w,:] = img
    return bg

def postprocess(result,ori_img,h_in,w_in,cut_bg,save_name = "result",re = True):
    h,w,_ = ori_img.shape

    classMap_numpy = result[0].argmax(0)
    classMap_numpy = classMap_numpy[:h_in, :w_in]
    mf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize=(15,15))
    classMap_numpy = cv2.morphologyEx(classMap_numpy.astype(np.uint8),cv2.MORPH_CLOSE,mf)
    classMap_numpy = cv2.morphologyEx(classMap_numpy.astype(np.uint8),cv2.MORPH_OPEN,mf)

    idx_fg = (classMap_numpy == 1).astype(np.uint8) * 255
    idx_fg = cv2.GaussianBlur(idx_fg, (5, 5), sigmaX=4)
    if re:
        idx_fg = cv2.resize(idx_fg, (w, h))
    idx_fg = np.expand_dims(idx_fg, 2).astype(np.float32) / 255

    seg_img = ori_img * idx_fg + cut_bg * (1 - idx_fg)
    cv2.imwrite("./{}.jpg".format(save_name),seg_img)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default='./test.png',
                        help='image for test')
    parser.add_argument('--onnx_model', type=str, default='./Dnc_SINet_bi_192_288.onnx',
                        help='onnx model')
    parser.add_argument('--mnn_model', type=str, default='./Dnc_SINet_bi_192_288.mnn',
                        help='mnn model')
    args = parser.parse_args()


    ######################################################################

    # initial onnxruntime
    import onnxruntime as ort
    from time import time
    from onnxruntime.capi import _pybind_state as C
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    print(args.onnx_model)
    ort_sess = ort.InferenceSession(args.onnx_model,sess_options=so)
    print(C.get_available_providers())
    ort_sess.set_providers(["CPUExecutionProvider"])


    # initial MNN
    import MNN
    interpreter = MNN.Interpreter(args.mnn_model)
    session = interpreter.createSession({"numThread": 1})
    input_tensor = interpreter.getSessionInput(session)
    ######################################################################

    _,_,model_in_h,model_in_w = ort_sess.get_inputs()[0].shape
    out_shape = ort_sess.get_outputs()[0].shape


    mean = [107.304565, 115.69884, 132.35703]
    std = [63.97182, 65.1337, 68.29726]

    # load test image

    ori_img = cv2.imread(args.image)
    input_img = ori_img.copy()
    input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    input_img = (input_img - mean) / std / 255
    input_img = input_img.astype(np.float32)
    h, w, _ = input_img.shape
    h_in, w_in = h, w
    re = False
    if h > model_in_h or w > model_in_w:
        re = True
        if h / w >= model_in_h / model_in_w:
            h_in = model_in_h
            w_in = int(w * (model_in_h / h))
        else:
            w_in = model_in_w
            h_in = int(h * (model_in_w / w))
    input_img = cv2.resize(input_img, (w_in, h_in))
    input_img = pad_32(input_img, expect_h=model_in_h, expect_w=model_in_w)
    input_img = input_img.transpose((2, 0, 1))
    input_img = np.expand_dims(input_img, 0)
    input_img = input_img.astype(np.float32)

    print(input_img.dtype)
    print(input_img.shape)
    syn_bg = cv2.imread("./bg.jpg")
    bg_h, bg_w, _ = syn_bg.shape
    center_y = bg_h // 2
    center_x = bg_w // 2
    cut_bg = syn_bg[center_y - h // 2:center_y + (h - h // 2), center_x - w // 2:center_x + (w - w // 2)]

    ###########################################################
    print("infer onnx...")
    # infer correction test
    onnx_out = ort_sess.run(None, {'input.1': input_img})

    print("infer MNN...")
    input_tensor.copyFrom(MNN.Tensor((1, 3, model_in_h,model_in_w), MNN.Halide_Type_Float,
                           input_img, MNN.Tensor_DimensionType_Caffe))
    interpreter.runSession(session)

    print("MNN output:")
    result = np.array(interpreter.getSessionOutput(session).getData())
    result = (np.reshape(result,(-1,4))[:,:2]).T.reshape(out_shape)
    print(result)
    print()
    postprocess(result,ori_img=ori_img,h_in=h_in,w_in=w_in,cut_bg=cut_bg,save_name="re_mnn")

    print("ONNX_output")
    print(onnx_out)
    print()
    postprocess(np.array(onnx_out[0]),ori_img=ori_img,h_in=h_in,w_in=w_in,cut_bg=cut_bg,save_name="re_onnx")





    ######################################################################
    print("running MNN...")
    s = time()
    for _ in range(1000):
        input_tensor.copyFrom(MNN.Tensor((1, 3, model_in_h,model_in_w), MNN.Halide_Type_Float,
                                         input_img, MNN.Tensor_DimensionType_Caffe))
        interpreter.runSession(session)
    e = time()
    print("MNN average runtime:{}".format((e-s)/1000))


    print("running onnxruntime...")
    s = time()
    for _ in range(1000):
        onnx_out = ort_sess.run(None, {'input.1':input_img})
    e = time()
    print("ort average runtime:{}".format((e-s)/1000))





