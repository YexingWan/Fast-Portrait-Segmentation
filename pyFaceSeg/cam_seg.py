import cv2
import numpy as np
import MNN
from argparse import ArgumentParser
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(1)

clicked = False
mean = [107.304565, 115.69884, 132.35703]
std = [63.97182, 65.1337, 68.29726]


def autocrop_row(image, threshold=0):
    """Crops any edges below or equal to threshold
    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        flatImage = image

    rows = np.where(np.max(flatImage,1) > threshold)[0]
    if rows.size:
        image = image[rows[0]: rows[-1] + 1,:]

    return image

def initial_model(model_path):
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession({"numThread": 2})
    input_tensor = interpreter.getSessionInput(session)
    output_tensor = interpreter.getSessionOutput(session)
    in_shape = input_tensor.getShape()
    out_shape = output_tensor.getShape()
    return interpreter,session, input_tensor,in_shape,out_shape

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

def onMouse(event,x,y,flags,param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

def preprocess(ori_img,model_in_h,model_in_w):
    # input_img = cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    input_img = ori_img.astype(np.float32)
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
    input_img = (input_img - mean) / std / 255
    input_img = pad_32(input_img, expect_h=model_in_h, expect_w=model_in_w)
    input_img = input_img.transpose((2, 0, 1))
    input_img = np.expand_dims(input_img, 0)
    input_img = input_img.astype(np.float32)
    return input_img,h_in,w_in,re


def postprocess(result, ori_img, h_in, w_in, cut_bg, re=True):
    h, w, _ = ori_img.shape
    result[0][0,:,:] += 0.814
    classMap_numpy = result[0].argmax(0)
    classMap_numpy = classMap_numpy[:h_in, :w_in]
    mf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3,3))
    classMap_numpy = cv2.morphologyEx(classMap_numpy.astype(np.uint8), cv2.MORPH_CLOSE, mf)
    classMap_numpy = cv2.morphologyEx(classMap_numpy.astype(np.uint8), cv2.MORPH_OPEN, mf)


    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(classMap_numpy)
    max_area = -1
    max_idx = -1

    if retval > 1:
        for m in range(1,retval):
            if stats[m][cv2.CC_STAT_AREA] > max_area:
                max_area = stats[m][cv2.CC_STAT_AREA]
                max_idx = m

        classMap_numpy = labels == max_idx

    idx_fg = (classMap_numpy == 1).astype(np.uint8) * 255
    idx_fg = cv2.GaussianBlur(idx_fg, (3, 3), sigmaX=2).astype(np.float32) / 255
    if re:
        idx_fg = cv2.resize(idx_fg, (w, h))
    idx_fg = np.expand_dims(idx_fg, 2)
    seg_img = ori_img * idx_fg + cut_bg * (1 - idx_fg)

    return seg_img

parser = ArgumentParser()


parser.add_argument('--model', type=str, default='./Dnc_SINet_bi_192_128.mnn',
                    help='mnn model')
args = parser.parse_args()


if __name__ == "__main__":

    cameraCapture = cv2.VideoCapture(0)
    cv2.namedWindow('seg_demo')
    cv2.setMouseCallback('seg_demo', onMouse)
    print('Showing camera feed.Click window or press any key to stop.')
    success, frame = cameraCapture.read()
    if not success:
        print("Fail to open camera.")
        exit(0)
    frame = autocrop_row(frame)

    frame_h,frame_w,_ = frame.shape


    # process background
    syn_bg = cv2.imread("./bg.jpg")
    bg_h, bg_w, _ = syn_bg.shape

    if bg_h / bg_w >= frame_h / frame_w:
        re_bg_w = int(frame_w+4)
        re_bg_h = int(bg_h * ((frame_w+4)/bg_w))
    else:
        re_bg_h = int(frame_h+4)
        re_bg_w = int(bg_w * ((frame_h+4)/bg_h))
    syn_bg = cv2.resize(syn_bg,(re_bg_w,re_bg_h))
    bg_h, bg_w, _ = syn_bg.shape
    center_y = bg_h // 2
    center_x = bg_w // 2
    cut_bg = syn_bg[center_y - frame_h // 2:center_y + (frame_h - frame_h // 2), center_x - frame_w // 2:center_x + (frame_w - frame_w // 2)]
    print("background shape:{}".format(syn_bg.shape))


    interpreter, session, input_tensor,input_shape,output_shape = initial_model(args.model)
    _,_,model_in_h, model_in_w = input_shape
    success = True


    while success and cv2.waitKey(1) == -1 and not clicked:
        success, frame = cameraCapture.read()
        # frame = autocrop_row(frame)
        if not success: break

        model_input,h_in,w_in,re = preprocess(frame,model_in_h,model_in_w)
        input_tensor.copyFrom(MNN.Tensor((1, 3, model_in_h,model_in_w), MNN.Halide_Type_Float,
                                         model_input, MNN.Tensor_DimensionType_Caffe))
        interpreter.runSession(session)
        result = np.array(interpreter.getSessionOutput(session).getData())
        result = (np.reshape(result, (-1, 4))[:, :2]).T.reshape(output_shape)
        seg_img = postprocess(result,ori_img=frame,h_in=h_in,w_in=w_in,cut_bg=cut_bg,re = re)
        cv2.imshow('seg_demo', np.fliplr(seg_img.astype(np.uint8)))
        #sleep(0.02)


    cv2.destroyWindow('seg_demo')
    cameraCapture.release()



