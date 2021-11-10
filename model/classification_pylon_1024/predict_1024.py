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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(f'{BASE_DIR}\\threshold.json'):
    print(f'{BASE_DIR}\\threshold.json', True)
else:
    print(BASE_DIR)

threshold_df = pd.DataFrame({
   "G-Mean":{
      "No Finding":0.3985043764,
      "Mass":0.0340008028,
      "Nodule":0.1290854067,
      "Lung Opacity":0.2623924017,
      "Patchy Opacity":0.0710924342,
      "Reticular Opacity":0.0751557872,
      "Reticulonodular Opacity":0.0219142251,
      "Nodular Opacity":0.0687219054,
      "Linear Opacity":0.0146626765,
      "Nipple Shadow":0.0109776305,
      "Osteoporosis":0.0194494389,
      "Osteopenia":0.0086441701,
      "Osteolytic Lesion":0.0037945583,
      "Fracture":0.0164119676,
      "Healed Fracture":0.0057689156,
      "Old Fracture":0.0044114138,
      "Spondylosis":0.135582611,
      "Scoliosis":0.0706555173,
      "Sclerotic Lesion":0.0074484712,
      "Mediastinal Mass":0.0034820705,
      "Cardiomegaly":0.3150766492,
      "Pleural Effusion":0.1500810385,
      "Pleural Thickening":0.0734574422,
      "Edema":0.029965058,
      "Hiatal Hernia":0.0021343548,
      "Pneumothorax":0.0055564633,
      "Atelectasis":0.0520619489,
      "Subsegmental Atelectasis":0.0070978594,
      "Elevation Of Hemidiaphragm":0.0301435925,
      "Tracheal-Mediastinal Shift":0.0052830973,
      "Volume Loss":0.0222865231,
      "Bronchiectasis":0.0110870562,
      "Enlarged Hilum":0.0042810268,
      "Atherosclerosis":0.0957187414,
      "Tortuous Aorta":0.0473294221,
      "Calcified Tortuous Aorta":0.0337250046,
      "Calcified Aorta":0.0314028598,
      "Support Devices":0.0634697229,
      "Surgical Material":0.0563883446,
      "Suboptimal Inspiration":0.0176297091
   },
   "Specificity":{
      "No Finding":0.6298416257,
      "Mass":0.0320259668,
      "Nodule":0.2103479952,
      "Lung Opacity":0.4027142823,
      "Patchy Opacity":0.1059797034,
      "Reticular Opacity":0.1644626409,
      "Reticulonodular Opacity":0.0535987988,
      "Nodular Opacity":0.1497070044,
      "Linear Opacity":0.028906459,
      "Nipple Shadow":0.0154532613,
      "Osteoporosis":0.0308215935,
      "Osteopenia":0.0201199837,
      "Osteolytic Lesion":0.0037945583,
      "Fracture":0.0217748992,
      "Healed Fracture":0.0057689156,
      "Old Fracture":0.00638283,
      "Spondylosis":0.2698899209,
      "Scoliosis":0.1694855392,
      "Sclerotic Lesion":0.010799651,
      "Mediastinal Mass":0.0067098509,
      "Cardiomegaly":0.4169337153,
      "Pleural Effusion":0.0582522154,
      "Pleural Thickening":0.180507049,
      "Edema":0.0213196687,
      "Hiatal Hernia":0.0012388981,
      "Pneumothorax":0.0047073471,
      "Atelectasis":0.1233060583,
      "Subsegmental Atelectasis":0.0180525146,
      "Elevation Of Hemidiaphragm":0.0420795679,
      "Tracheal-Mediastinal Shift":0.0050427029,
      "Volume Loss":0.015577767,
      "Bronchiectasis":0.0159820151,
      "Enlarged Hilum":0.0090267183,
      "Atherosclerosis":0.4578576684,
      "Tortuous Aorta":0.1591946483,
      "Calcified Tortuous Aorta":0.1176451445,
      "Calcified Aorta":0.1489057392,
      "Support Devices":0.0483015031,
      "Surgical Material":0.0178405829,
      "Suboptimal Inspiration":0.0432249047
   },
   "Sensitivity":{
      "No Finding":0.2152858078,
      "Mass":0.0067230272,
      "Nodule":0.0733733624,
      "Lung Opacity":0.1852027774,
      "Patchy Opacity":0.0592248365,
      "Reticular Opacity":0.0600548796,
      "Reticulonodular Opacity":0.0226954054,
      "Nodular Opacity":0.0500446856,
      "Linear Opacity":0.0073502855,
      "Nipple Shadow":0.0063510253,
      "Osteoporosis":0.015578717,
      "Osteopenia":0.0001913563,
      "Osteolytic Lesion":0.0031488203,
      "Fracture":0.0086568585,
      "Healed Fracture":0.0052975286,
      "Old Fracture":0.003852841,
      "Spondylosis":0.1176342815,
      "Scoliosis":0.0431854539,
      "Sclerotic Lesion":0.0043818625,
      "Mediastinal Mass":0.0012249933,
      "Cardiomegaly":0.3056172431,
      "Pleural Effusion":0.2747670114,
      "Pleural Thickening":0.0511276089,
      "Edema":0.0488492139,
      "Hiatal Hernia":0.0001794333,
      "Pneumothorax":0.0055564633,
      "Atelectasis":0.0348859914,
      "Subsegmental Atelectasis":0.008214497,
      "Elevation Of Hemidiaphragm":0.0285350438,
      "Tracheal-Mediastinal Shift":0.0106371948,
      "Volume Loss":0.0412723497,
      "Bronchiectasis":0.0119529469,
      "Enlarged Hilum":0.0024274809,
      "Atherosclerosis":0.0835555047,
      "Tortuous Aorta":0.0290249493,
      "Calcified Tortuous Aorta":0.031879209,
      "Calcified Aorta":0.0278305244,
      "Support Devices":0.0536448546,
      "Surgical Material":0.261854589,
      "Suboptimal Inspiration":0.0176297091
   }
})
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

    # risk_dict = {'risk_score': 1 - df_prob_class['No Finding'].values[0]}
    all_pred_df = get_all_pred_df(CATEGORIES, y_calibrated, y_uncalibrated, threshold_dict)

    return pred, seg, all_pred_class, all_pred_df

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

async def main(file_location, content_type):
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
    pred, seg, all_pred_class, all_pred_df = predict(image, net_predict, threshold_dict, class_dict)
    
    path = os.path.join(os.path.dirname(file_location), 'result')
    if not os.path.exists(path):
        os.makedirs(path)
    
    if ('dicom' in str(content_type) or 'octet-stream' in str(content_type)) and  image_load.shape[0] > 1500:
        scale = 1024/image_load.shape[0]
        image_load = cv2.resize(image_load, (0,0), fx=scale, fy=scale)

    print('save images')
    for i_class in important_finding:
        cam = overlay_cam(np.array(image_load), seg[0, class_dict[i_class]])

        scale = 512/cam.shape[0]
        cam = cv2.cvtColor(np.float32(cam*255), cv2.COLOR_BGR2RGB)
        cam = cv2.resize(cam, (0,0), fx=scale, fy=scale)
        cv2.imwrite(os.path.join(path, i_class + '.png'), cam)

    return all_pred_df
