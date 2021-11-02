from model.classification_pylon_1024.dataset_cpu import *

import onnxruntime

import PIL
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

import numpy as np

import os

important_finding = focusing_finding[1:]
BASE_DIR = os.path.dirname(__file__)

threshold_df = pd.read_json(f'{BASE_DIR}\\threshold.json')
threshold_dict = threshold_df['G-Mean'].to_dict()
CATEGORIES = list(threshold_dict.keys())
class_dict = {cls:i for i, cls in enumerate(CATEGORIES)}


# Convert DICOM to Numpy Array
def dicom2array(path, voi_lut=True, fix_monochrome=True):
    """Convert DICOM file to numy array

    Args:
        path (str): Path to the DICOM file to be converted
        voi_lut (bool): Whether or not VOI LUT is available
        fix_monochrome (bool): Whether or not to apply MONOCHROME fix

    Returns:
        Numpy array of the respective DICOM file
    """

    # Use the pydicom library to read the DICOM file
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # Depending on this value, X-ray may look inverted - fix that
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # Normalize the image array
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return dicom, data

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
        result_dict['Threshold'].append(float(threshold_dict[pred_cls]))
        result_dict['Raw_Pred'].append(float(uncalibrated_prob))
        result_dict['Confidence'].append(float(calibrated_prob))
        result_dict['isPositive'].append(bool(uncalibrated_prob>=threshold_dict[pred_cls]))

    # all_pred_df = pd.DataFrame(result_dict)
    return result_dict

def sigmoid_array(x):
    # if x >= 0:
    #     return 1 / (1 + np.exp(-x)) >> RuntimeWarning: overflow encountered in exp
    # else:
    return np.exp(x)/(1+np.exp(x))

def preprocess(image):
    _res = eval_transform(image=np.array(image))
    image = np.float32(_res)
    image = np.expand_dims(image, axis=(0, 1))
    return image

def predict(image, net_predict, threshold_dict, class_dict):
    ort_session = net_predict
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    out = ort_session.run(None, ort_inputs)
    pred, seg = out
    # pred = torch.from_numpy(pred)
    # seg= torch.from_numpy(seg)
    pred, seg = sigmoid_array(pred), sigmoid_array(seg)

    # Interpolation to target size
    width = 1024
    height = 1024

    img_stack = seg[0]
    img_stack_sm = np.zeros((len(img_stack), width, height))

    for idx in range(len(img_stack)):
        img = img_stack[idx, :, :]
        img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        img_stack_sm[idx, :, :] = img_sm

    seg = np.expand_dims(img_stack_sm, axis=0)

    y_pred = pred.copy()
    y_calibrated = pred.copy()
    y_uncalibrated = pred.copy()

    for i, (c_name, thredshold) in enumerate(threshold_dict.items()):
        i_class = class_dict[c_name]
        # y_calibrated[:,i_class] = calibrate_prob(y_calibrated, pred_val, c_name)/100
        # y_calibrated[:,i_class] = np.clip(y_calibrated[:,i_class]/temperature_dict[c_name], 0.00, 1.00) #  calibrating prob with weight from temperature scaling technique
        y_pred[:, i_class][y_pred[:, i_class] >= thredshold] = 1
        y_pred[:, i_class][y_pred[:, i_class] < thredshold] = 0

    all_pred_class = np.array(CATEGORIES)[y_pred[0] == 1]

    df_prob_class = pd.DataFrame(y_uncalibrated, columns=CATEGORIES)  # To use risk score from raw value of model

    risk_dict = {'risk_score': 1 - df_prob_class['No Finding'].values[0]}
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
    x = x - x.min()
    x = x / x.max()
    # resize the cam
    x = cv2.resize(x, (w, h))
    x = x - x.min()
    x = x / x.max()
    # coloring the cam
    x = cv2.applyColorMap(np.uint8(255 * (1 - x)), cv2.COLORMAP_JET)
    x = np.float32(x) / 255.

    # overlay
    x = img / img_max + weight * x
    x = x / x.max()
    return x

def main(file_location, content_type):
    checkpoint = os.path.join(BASE_DIR, 'pylon_densenet169_ImageNet_1024_selectRad_V2.onnx')

    size=1024
    interpolation='cubic'
    dev='cpu'

    net_predict = onnxruntime.InferenceSession(checkpoint)
    image_load = ''

    if 'dicom' in str(content_type) or 'octet-stream' in str(content_type):
        dicom, data = dicom2array(file_location)
        image_load = data
    elif 'png' in str(content_type):
        image_load = PIL.Image.open(file_location)

    image = preprocess(image_load)
    pred, seg, all_pred_class, all_pred_df,risk_dict = predict(image, net_predict, threshold_dict, class_dict)
    
    os.makedirs(os.path.join(os.path.dirname(file_location), 'result'), exist_ok=True)
    
    path = os.path.join(os.path.dirname(file_location), 'result')
    for i_class in important_finding:
        cam = overlay_cam(np.array(image_load), seg[0, class_dict[i_class]])

        cam = cv2.cvtColor(np.float32(cam*255), cv2.COLOR_BGR2RGB)
        cam = cv2.resize(cam, (0,0), fx=0.5, fy=0.5)
        cv2.imwrite(path + '\\' + i_class + '.png', cam)

    return all_pred_df
