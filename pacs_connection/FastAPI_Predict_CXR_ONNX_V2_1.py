# Basic library
import math
import os
import numpy as np
import pickle
from io import BytesIO
import gc

# Statistic library
from scipy import stats

# Image preprocessing
import cv2
import PIL
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# Deep learning library for pre&post processing
import GPUtil
# import torch.nn.functional as F
# from torch import Tensor
# from torch.autograd import Variable
import onnxruntime

# Local library
from dataset import *
from Utilis_DICOM import dicom2array, resize_image

# checkpoint = 'save/pylon_densenet169_ImageNet_1024/0/best'
name = 'pylon_densenet169_ImageNet_1024_selectRad_V2'
checkpoint = f'save/onnx/{name}.onnx'
# checkpoint_flush = f'save/onnx/{name}_flush.onnx'

size=1024
interpolation='cubic'

# setting device on GPU if available, else CPU
# https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
dev = 'cuda' if GPUtil.getAvailable() else 'cpu'
print('Using device:', dev)
print()
#Additional Info when using cuda
# if dev.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

print('onnxruntime device:', onnxruntime.get_device())

# threshold_df = pd.read_json(f'./save/threshold/{name}_threshold.json')
threshold_df = pd.read_json(f'./save/threshold/{name}_combine_threshold.json')
threshold_dict = threshold_df['G-Mean'].to_dict()
# threshold_dict = threshold_df['F1_Score'].to_dict()
CATEGORIES = list(threshold_dict.keys())
class_dict = {cls:i for i, cls in enumerate(CATEGORIES)}
# df_json = pd.read_json('./save/temperature_parameter.json')
# temperature_dict = df_json['Temperature'].to_dict()

# Val prediction for make percentile
# with open('save/val_predict/out_val_selectRad_data_pylon_densenet169_ImageNet_1024_V2_0.p', 'rb') as fp:
#     val_predict = pickle.load(fp)
# pred_val = val_predict['pred']

# temperature scaling weight
# temperature_dict = pd.read_json('save/temperature_weight.json').set_index('Finding')['Temperature'].to_dict()

focusing_finding = [
                    # 'No Finding', 
                    'Pneumothorax', 
#                     'Pneumomediastinum', 'Pneumopericardium',
                    'Mass', 'Nodule', 'Mediastinal Mass', 
                    'Lung Opacity', 'Pleural Effusion', 
                    'Atelectasis', 
#                     'Airway Narrowing', 
                    'Tracheal-Mediastinal Shift', 'Volume Loss', 
                    'Osteolytic Lesion', 'Fracture', 'Sclerotic Lesion', 
#                     'Air-filled Space', 
                    'Cardiomegaly', 
                    'Bronchiectasis',
#                     'Increased Lung Volume'
                   ]
focusing_finding_dict = {cls:i for i, cls in enumerate(focusing_finding)}

def sigmoid_array(x):
    # if x >= 0:
    #     return 1 / (1 + np.exp(-x)) >> RuntimeWarning: overflow encountered in exp
    # else:
    return np.exp(x)/(1+np.exp(x))                             

def read_imagefile(file) -> PIL.Image.Image:
    image = PIL.Image.open(BytesIO(file))
    return image

def preprocess(image):
    _res = eval_transform(image=np.array(image))
    image = np.float32(_res)
    # image = Variable(image, requires_grad=False)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    image = np.expand_dims(image, axis=(0, 1))
    return image

net_predict = onnxruntime.InferenceSession(checkpoint)

def get_all_pred_df(CATEGORIES, y_calibrated, y_uncalibrated, threshold_dict):
    result_dict = {}
    result_dict['Finding'] = []
    result_dict['Threshold'] = []
    result_dict['Raw_Pred'] = []
    result_dict['Confidence'] = []
    result_dict['isPositive'] = []
    # for pred_cls, prob in zip(all_pred_class, all_prob_class):
    for pred_cls, calibrated_prob, uncalibrated_prob in zip(CATEGORIES, np.array(y_calibrated.ravel()), np.array(y_uncalibrated.ravel())):
        result_dict['Finding'].append(pred_cls)
        result_dict['Threshold'].append(threshold_dict[pred_cls])
        result_dict['Raw_Pred'].append(uncalibrated_prob)
        result_dict['Confidence'].append(calibrated_prob)
        result_dict['isPositive'].append(bool(uncalibrated_prob>=threshold_dict[pred_cls]))

    all_pred_df = pd.DataFrame(result_dict)
    return all_pred_df

# จากปัญหาเรื่อง overconfidence เลยจำเป็นต้อง manual post process แบบนี้ คือ กำหนดให้ค่า pred_prob ที่ threshold เป็น pctile ที่ 50 ไปเลย
# แต่คำนวณ ECE ออกมาแย่มาก เลยกลับไปใช้ Temperature scaling แทน
def calibrate_prob(pred, pred_val, c_name, kind='rank'):
    thredshold = threshold_dict[c_name]
    i_class = class_dict[c_name]

    pred_val_c_name = pred_val[:,i_class]
    y_pred_val = np.array(pred_val).copy()

    is_over_threshold = pred[:,i_class] >= thredshold
    if is_over_threshold:
        y_pred_val[:,i_class][y_pred_val[:,i_class] >= thredshold] = 1
        y_pred_val[:,i_class][y_pred_val[:,i_class] < thredshold] = 0
        y_pred_bool = y_pred_val[:,i_class].astype(bool)
        pred_array_sel_threshold = pred_val_c_name[y_pred_bool]
        result = stats.percentileofscore(pred_array_sel_threshold, pred[:,i_class], kind=kind)
        result = 50 + result/2
    else:
        y_pred_val[:,i_class][y_pred_val[:,i_class] >= thredshold] = 0
        y_pred_val[:,i_class][y_pred_val[:,i_class] < thredshold] = 1
        y_pred_bool = y_pred_val[:,i_class].astype(bool)
        pred_array_sel_threshold = pred_val_c_name[y_pred_bool]
        result = stats.percentileofscore(pred_array_sel_threshold, pred[:,i_class], kind=kind)
        result = result/2
    # print('Min value of this class:',min(pred_array_over_threshold), 
    #       '\nMax value of this class:', max(pred_array_over_threshold))
    # print(pred[:,i_class])
    
    return result

def predict(image, net_predict, threshold_dict, class_dict):
    
    ort_session = net_predict
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    out = ort_session.run(None, ort_inputs)
    pred ,seg = out
    # pred = torch.from_numpy(pred)
    # seg= torch.from_numpy(seg)
    pred, seg = sigmoid_array(pred), sigmoid_array(seg)

    # Interpolation
    # seg = cv2.resize(seg, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    width = 1024
    height = 1024

    img_stack = seg[0]
    img_stack_sm = np.zeros((len(img_stack), width, height))

    for idx in range(len(img_stack)):
        img = img_stack[idx, :, :]
        img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        img_stack_sm[idx, :, :] = img_sm
        
    seg = np.expand_dims(img_stack_sm, axis=0)

    del ort_session
    del net_predict
    del out
    gc.collect()
    
    y_pred = pred.copy()
    y_calibrated = pred.copy()
    y_uncalibrated = pred.copy()

    for i, (c_name, thredshold) in enumerate(threshold_dict.items()):
        i_class = class_dict[c_name]
        # y_calibrated[:,i_class] = calibrate_prob(y_calibrated, pred_val, c_name)/100
        # y_calibrated[:,i_class] = np.clip(y_calibrated[:,i_class]/temperature_dict[c_name], 0.00, 1.00) #  calibrating prob with weight from temperature scaling technique
        y_pred[:,i_class][y_pred[:,i_class] >= thredshold] = 1
        y_pred[:,i_class][y_pred[:,i_class] < thredshold] = 0
        
        
    all_pred_class = np.array(CATEGORIES)[y_pred[0] == 1]

    df_prob_class = pd.DataFrame(y_uncalibrated, columns=CATEGORIES) # To use risk score from raw value of model

    risk_dict = {'risk_score': 1-df_prob_class['No Finding'].values[0]}
    all_pred_df = get_all_pred_df(CATEGORIES, y_calibrated, y_uncalibrated, threshold_dict)
    
    return pred, seg, all_pred_class, all_pred_df, risk_dict

def overlay_cam(img, cam, weight=0.5, img_max=255.):
    """
    Red is the most important region
    Args:
        img: numpy array (h, w) or (h, w, 3)
    """
    if len(img.shape) == 2:
        h, w = img.shape
        img = img.reshape(h, w, 1)
        img = np.repeat(img, 3, axis=2)

    # print('img:',img.shape)
    # print('seg:',cam.shape)
    h, w, c = img.shape

    img_max = img.max()
    # normalize the cam
    x = cam
    # x = x - x.min()
    # x = x / x.max()
    # resize the cam
    x = cv2.resize(x, (w, h))
    # x = x - x.min()
    # x = x / x.max()
    # Clip value to [0 1]
    x = np.clip(x, 0, 1)

    # coloring the cam
    x = cv2.applyColorMap(np.uint8(255 * (1 - x)), cv2.COLORMAP_JET)
    x = np.float32(x) / 255.

    # overlay
    x = img / img_max + weight * x
    x = x / x.max()
    return x


def plot_bbox_from_df(df_bbox, image_path):

    ### df_bbox คือ json ที่ได้มาจาก database
    ### image_path คือ path ของ image

    import cv2
    import pydicom
    import numpy as np
    import pandas as pd

    all_bbox = pd.DataFrame(df_bbox['data'], columns=['label', 'tool', 'data'])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    thickness = 6
    isClosed = True
    # Green color
    color = (0, 255, 0)

    ### ส่วนที่เรียก image
    #import dicom image array
    if image_path != None:
        dicom = pydicom.dcmread(("img_path"))
        inputImage = dicom.pixel_array

        # depending on this value, X-ray may look inverted - fix that:
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            inputImage = np.amax(inputImage) - inputImage
            inputImage = np.stack([inputImage, inputImage, inputImage])
            inputImage = inputImage.astype('float32')
            inputImage = inputImage - inputImage.min()
            inputImage = inputImage / inputImage.max()
            inputImage = inputImage.transpose(1, 2, 0)
            inputImage = (inputImage*255).astype(int)
            # https://github.com/opencv/opencv/issues/14866
            inputImage = cv2.UMat(inputImage).get()
    else:
        inputImage = np.zeros([3000, 3000, 3])
        inputImage = inputImage.astype(int)
        inputImage = cv2.UMat(inputImage).get()

    for index, row in all_bbox.iterrows():

        if row['tool'] == 'rectangleRoi':
            pts = row["data"]["handles"]
            inputImage = cv2.rectangle(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
                pts["end"]["x"]), int(pts["end"]["y"])), color, thickness)
            inputImage = cv2.putText(inputImage,  row["label"], (int(min(pts["start"]["x"], pts["end"]["x"])), int(
                min(pts["start"]["y"], pts["end"]["y"]))), font, fontScale, color, thickness, cv2.LINE_AA)

        if row['tool'] == 'freehand':
            pts = np.array([[cdn["x"], cdn["y"]]
                           for cdn in row["data"]["handles"]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            #choose min x,y for text origin
            text_org = np.amin(pts, axis=0)
            inputImage = cv2.polylines(inputImage, [pts], isClosed, color, thickness)
            inputImage = cv2.putText(inputImage,  row["label"], tuple(
                text_org[0]), font, fontScale, color, thickness, cv2.LINE_AA)

        if row['tool'] == 'length':
            pts = row["data"]["handles"]
            #choose left point for text origin
            text_org = "start"
            if pts["start"]["x"] > pts["end"]["x"]:
                text_org = "end"
            inputImage = cv2.line(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
                pts["end"]["x"]), int(pts["end"]["y"])), color, thickness)
            inputImage = cv2.putText(inputImage,  row["label"], (int(pts[text_org]["x"]), int(
                pts[text_org]["y"])), font, fontScale, color, thickness, cv2.LINE_AA)

        if row['tool'] == 'ratio':
            pts = row["data"]["0"]["handles"]
            #choose left point for text origin
            text_org = "start"
            if pts["start"]["x"] > pts["end"]["x"]:
                text_org = "end"
            inputImage = cv2.line(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
                pts["end"]["x"]), int(pts["end"]["y"])), color, thickness)
            inputImage = cv2.putText(inputImage,  row["label"], (int(pts[text_org]["x"]), int(
                pts[text_org]["y"])), font, fontScale, color, thickness, cv2.LINE_AA)
            pts = row["data"]["1"]["handles"]
            #choose left point for text origin
            text_org = "start"
            if pts["start"]["x"] > pts["end"]["x"]:
                text_org = "end"
            inputImage = cv2.line(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
                pts["end"]["x"]), int(pts["end"]["y"])), color, thickness)
            inputImage = cv2.putText(inputImage,  row["label"], (int(pts[text_org]["x"]), int(
                pts[text_org]["y"])), font, fontScale, color, thickness, cv2.LINE_AA)

        #cv2.imshow(inputImage)
        #cv2.imwrite("/path/to/save.png", inputImage)
    return inputImage
        
