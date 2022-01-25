from io import BytesIO
from model.classification_pylon.dataset_cpu import *

import onnxruntime

import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

import numpy as np

import os

import json
import traceback

important_finding = focusing_finding[1:]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

threshold_df = pd.DataFrame({
   "G-Mean":{
      "No Finding":0.3972887695,
      "Mass":0.0609776899,
      "Nodule":0.230503425,
      "Lung Opacity":0.292879045,
      "Patchy Opacity":0.0865180939,
      "Reticular Opacity":0.0958138555,
      "Reticulonodular Opacity":0.0563067533,
      "Nodular Opacity":0.1255027354,
      "Linear Opacity":0.0148700876,
      "Nipple Shadow":0.0144643001,
      "Osteoporosis":0.0195371378,
      "Osteopenia":0.008630353,
      "Osteolytic Lesion":0.1086166054,
      "Fracture":0.1394902021,
      "Healed Fracture":0.0305878147,
      "Old Fracture":0.0043833959,
      "Spondylosis":0.1802854091,
      "Scoliosis":0.1115784869,
      "Sclerotic Lesion":0.0159624852,
      "Mediastinal Mass":0.009116156,
      "Cardiomegaly":0.3053534329,
      "Pleural Effusion":0.0764062479,
      "Pleural Thickening":0.1011081636,
      "Edema":0.029946018,
      "Hiatal Hernia":0.0055465782,
      "Pneumothorax":0.0046908478,
      "Atelectasis":0.1262948215,
      "Subsegmental Atelectasis":0.0099582328,
      "Elevation Of Hemidiaphragm":0.033518523,
      "Tracheal-Mediastinal Shift":0.0094351135,
      "Volume Loss":0.0116786687,
      "Bronchiectasis":0.0099655734,
      "Enlarged Hilum":0.0124651333,
      "Atherosclerosis":0.1975888014,
      "Tortuous Aorta":0.0471370481,
      "Calcified Tortuous Aorta":0.0337999463,
      "Calcified Aorta":0.0313448533,
      "Support Devices":0.0370221362,
      "Surgical Material":0.056236092,
      "Suboptimal Inspiration":0.0176099241
   },
   "F1_Score":{
      "No Finding":0.6230675578,
      "Mass":0.3871906698,
      "Nodule":0.3973264992,
      "Lung Opacity":0.3817337453,
      "Patchy Opacity":0.1769620031,
      "Reticular Opacity":0.1895534098,
      "Reticulonodular Opacity":0.5747563839,
      "Nodular Opacity":0.6876279712,
      "Linear Opacity":0.3714343011,
      "Nipple Shadow":0.1233664379,
      "Osteoporosis":0.1537039429,
      "Osteopenia":0.0570114627,
      "Osteolytic Lesion":0.286719352,
      "Fracture":0.347702384,
      "Healed Fracture":0.6489098072,
      "Old Fracture":0.5180939436,
      "Spondylosis":0.3076757491,
      "Scoliosis":0.2380676866,
      "Sclerotic Lesion":0.3251270652,
      "Mediastinal Mass":0.4449528158,
      "Cardiomegaly":0.4099305868,
      "Pleural Effusion":0.170565486,
      "Pleural Thickening":0.1786464006,
      "Edema":0.2325405926,
      "Hiatal Hernia":0.1721060127,
      "Pneumothorax":0.2610811889,
      "Atelectasis":0.6366727352,
      "Subsegmental Atelectasis":0.0099582328,
      "Elevation Of Hemidiaphragm":0.0423872843,
      "Tracheal-Mediastinal Shift":0.565830946,
      "Volume Loss":0.6945303679,
      "Bronchiectasis":0.2106481194,
      "Enlarged Hilum":0.0333859064,
      "Atherosclerosis":0.1176939905,
      "Tortuous Aorta":0.3932620287,
      "Calcified Tortuous Aorta":0.3628055155,
      "Calcified Aorta":0.3663214743,
      "Support Devices":0.895901382,
      "Surgical Material":0.3150249124,
      "Suboptimal Inspiration":0.5532807112
   }
})
threshold_dict = threshold_df['F1_Score'].to_dict()
CATEGORIES = list(threshold_dict.keys())
class_dict = {cls:i for i, cls in enumerate(CATEGORIES)}


# Convert DICOM to Numpy Array
def dicom2array(file, voi_lut=True, fix_monochrome=True):
    """Convert DICOM file to numy array
    
    Args:
        file : input object or uploaded file
        path (str): Path to the DICOM file to be converted
        voi_lut (bool): Whether or not VOI LUT is available
        fix_monochrome (bool): Whether or not to apply MONOCHROME fix
        
    Returns:
        Numpy array of the respective DICOM file
    """
    
    # Use the pydicom library to read the DICOM file

    if not isinstance(file, (pydicom.FileDataset, pydicom.dataset.Dataset)):
        try: # If file is uploaded with fastapi.UploadFile
            path = BytesIO(file)
            dicom = pydicom.read_file(path)
        except: # If file is uploaded with streamlit.file_uploader
            dicom = pydicom.read_file(file)
    else: # if file is readed dicom file
        dicom = file
    
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

def preprocess(image, size = 1024):
    _res = eval_transform(image=np.array(image), size=size)
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

def resize_image(array, size, keep_ratio=False, resample=Image.LANCZOS):
    image = Image.fromarray(array)
    
    if keep_ratio:
        image.thumbnail((size, size), resample)
    else:
        image = image.resize((size, size), resample)
    
    return np.array(image)

def main(ds, file_dir, model_name):
    print("Inference start")
    try:
        checkpoint = ""
        model_size = '1024'
        # if model_name == "classification_pylon_256":
        #     model_size = '256'

        if model_size == '1024':
            checkpoint = os.path.join(BASE_DIR, 'pylon_densenet169_ImageNet_1024_selectRad_V2.onnx')
        elif model_size == '256':
            checkpoint = os.path.join(BASE_DIR, 'pylon_densenet169_ImageNet_256_selectRad_V2.onnx')

        net_predict = onnxruntime.InferenceSession(checkpoint)
        image_load = ''

        if isinstance(ds, (pydicom.FileDataset, pydicom.dataset.Dataset)):
            dicom, image_load = dicom2array(ds)
        else:
            image_load = ds

        image_load = resize_image(image_load, size=1024, keep_ratio=True)

        image = preprocess(image_load, int(model_size))
        pred, seg, all_pred_class, all_pred_df = predict(image, net_predict, threshold_dict, class_dict)
        
        res_dir = os.path.join(file_dir, 'result')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        
        if image_load.shape[0] > 1500:
            scale = 1024/image_load.shape[0]
            image_load = cv2.resize(image_load, (0,0), fx=scale, fy=scale)

        print('Begin saving heatmap process')
        scale = 512/np.array(image_load).shape[0]
        for i_class in important_finding:
            cam = overlay_cam(np.array(image_load), seg[0, class_dict[i_class]])

            cam = cv2.cvtColor(np.float32(cam*255), cv2.COLOR_BGR2RGB)
            cam = cv2.resize(cam, (0,0), fx=scale, fy=scale)
            cv2.imwrite(os.path.join(res_dir, i_class + '.png'), cam)

        original_image = cv2.resize(np.array(image_load), (0,0), fx=scale, fy=scale)
        cv2.imwrite(os.path.join(res_dir, 'original.png'), original_image)

        if 'PatientName' in ds:
            all_pred_df['patient_name'] = ds.PatientName.given_name + " " + ds.PatientName.family_name
        with open(os.path.join(res_dir, 'prediction.txt'), 'w') as f:
            json.dump(all_pred_df, f)
        
        return True, ""
    except Exception as e:
        print(traceback.format_exc())
        return False, e