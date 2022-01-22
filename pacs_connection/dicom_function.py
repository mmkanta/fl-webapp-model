import os, sys
import traceback
import numpy as np
import pandas as pd
import pickle
import fnmatch
import subprocess
import pydicom

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACS_DIR = os.path.join(BASE_DIR, 'resources', 'files')
TEMP_DIR = os.path.join(BASE_DIR, 'resources', 'temp')
LOCAL_DIR = os.path.join(BASE_DIR, 'resources', 'local')
sys.path.append(BASE_DIR)

from evt_classes import *
from Utilis_DICOM import array_to_dicom
from pathlib import Path
import time
import shutil

import gc

import argparse

import json

from model.classification_pylon.predict import main as pylon_predict
from model.covid19_admission.predict_admission import main as covid_predict

# https://zetcode.com/python/argparse/
parser = argparse.ArgumentParser()
   
parser.add_argument('-n', '--hn', type=str, default='', help="Patient's HN")
parser.add_argument('-f', '--function', type=str, default="", help="Function's Name")
parser.add_argument('-a', '--accession_no', type=str, default="", help="Accession Number")
parser.add_argument('-m', '--model', type=str, default="", help="Model's Name")
parser.add_argument('-b', '--bounding_box', type=str, default="", help="Bounding Box Dict")

args = parser.parse_args()

def load_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# dicom function called by /api
def get_all(hn):
    try:
        all_data = []
        for file in os.listdir(PACS_DIR):
            if file.endswith('.evt'):
                event = load_file(os.path.join(PACS_DIR, file))
                ds = event.dataset
                if ds.PatientID == hn:
                    data = dict()
                    data['Accession No'] = ds.AccessionNumber
                    data['Modality'] = ds.Modality
                    data['Patient ID'] = ds.PatientID
                    data['Patient Name'] = ds.PatientName.given_name + " " + ds.PatientName.family_name
                    data['Patient Sex'] = ds.PatientSex
                    if 'PatientAge' in ds:
                        data['Age'] = ds.PatientAge
                    try:
                        data['Procedure Code'] = ds[0x020,0x0010].value
                    except:
                        data["Procedure Code"] = ""
                    try:
                        data['Study Date Time'] = str(pd.to_datetime(ds.StudyDate)) # ds.StudyTime
                    except:
                        data["Study Date Time"] = ""
                    # data['Proc Description'] = 'Chest PA upright'
                    all_data.append(data)
        if all_data != []:
            with open(os.path.join(TEMP_DIR, "dicom_files_{}.json".format(hn)), 'w') as f:
                json.dump(all_data, f)
    except Exception as e:
        print(traceback.format_exc())

def get_info(hn):
    try:
        all_data = {}
        for file in os.listdir(PACS_DIR):
            if file.endswith('.evt'):
                event = load_file(os.path.join(PACS_DIR, file))
                ds = event.dataset
                if ds.PatientID == hn:
                    data = dict()
                    data['Patient ID'] = ds.PatientID
                    data['Patient Name'] =  ds.PatientName.given_name + " " + ds.PatientName.family_name
                    data['Patient Sex'] = ds.PatientSex
                    try:
                        data['Patient Birthdate'] = str(pd.to_datetime(ds.PatientBirthDate))
                    except:
                        data['Patient Birthdate'] = ds.PatientBirthDate
                    if 'PatientAge' in ds:
                        data['Age'] = ds.PatientAge
                    all_data = data
                    with open(os.path.join(TEMP_DIR, "dicom_info_{}.json".format(hn)), 'w') as f:
                        json.dump(all_data, f)
                    break
    except Exception as e:
        print(traceback.format_exc())

def get_dicom(acc_no):
    try:
        acc_no = str(acc_no)
        file_dir = os.path.join(TEMP_DIR, "{}_get".format(acc_no))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if os.path.exists(os.path.join(PACS_DIR, acc_no + '.evt')):
            event = load_file(os.path.join(PACS_DIR, acc_no + '.evt'))
            ds = event.dataset
            ds.file_meta = event.file_meta
            ds.save_as(os.path.join(file_dir, "{}_tmp.dcm".format(acc_no)))
            subprocess.run(["gdcmconv", "-C",  os.path.join(file_dir, "{}_tmp.dcm".format(acc_no)), os.path.join(file_dir, "{}.dcm".format(acc_no))])
        else:
            with open(os.path.join(file_dir, "not_found.txt"), 'w') as f:
                f.write('not found')
    except Exception as e:
        print(traceback.format_exc())

def infer(acc_no, model_name):
    try:
        acc_no = str(acc_no)
        file_dir = os.path.join(TEMP_DIR, "{}_{}".format(model_name, acc_no))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if os.path.exists(os.path.join(PACS_DIR, acc_no + '.evt')):
            event = load_file(os.path.join(PACS_DIR, acc_no + '.evt'))
            ds = event.dataset
            ds.file_meta = event.file_meta

            message = "Error occurred"
            result = False
            if "classification_pylon" in model_name:
                result, message = pylon_predict(ds, file_dir, model_name)
            # elif model_name == "covid19_admission":
            #     result = covid_predict(file_dir)
            else:
                message = "Model not found"

            if result:
                with open(os.path.join(file_dir, "success.txt"), 'w') as f:
                    f.write('infer successfully')
            else:
                with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
                    f.write(message)
        else:
            with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
                f.write('Cannot find dicom file')
    except Exception as e:
        print(traceback.format_exc())

def save_to_pacs(acc_no, bbox):
    # bbox string to dict
    bbox_dict = json.loads(bbox)
    bbox_heatmap_dir = os.path.join(BASE_DIR, 'resources', 'temp', acc_no + '_store')
    event = load_file(os.path.join(PACS_DIR, acc_no + '.evt'))

    ds = event.dataset
    ds.file_meta = event.file_meta

    Accession_Number = ds.AccessionNumber

    for file in os.listdir(bbox_heatmap_dir):
        filename = os.fsdecode(file)
        finding = filename.split('.')[0]
        ds_modify, dcm_compressed_path = array_to_dicom(ds, os.path.join(bbox_heatmap_dir), filename)
        if dcm_compressed_path is None:
            print('Cannot convert image {finding} to dicom')
            with open(os.path.join(bbox_heatmap_dir, "fail.txt"), 'w') as f:
                f.write('Cannot convert image {finding} to dicom')
            return

        # print(f'Receive DICOM and processing complete with execution time: {time.time() - start_time :.2f} seconds')  

        # SCU Role
        start_time = time.time()
        ae_title_scp = "SYNAPSEDICOM" #"AE_TITLE_NRT02" #   "MY_ECHO_SCP_AWS" # 
        addr = "192.1.10.200" #"192.1.10.162" #   "13.229.184.70" #
        port = 104 # 104 # 11114 #
        command = f"python SCU.py -a {ae_title_scp} -s {addr} -p {port} -f {dcm_compressed_path}"
        subprocess.run(command.split())

        
        print(f'Send {Accession_Number} Modified DICOM "{finding}" with execution time: {time.time() - start_time :.2f} seconds')
        print('  {finding} Done  '.center(100,'='))

    for k, val in bbox_dict['classes'].items():
        # bbox to dicom
        
        # SCU Role
        start_time = time.time()
        ae_title_scp = "SYNAPSEDICOM" #"AE_TITLE_NRT02" #   "MY_ECHO_SCP_AWS" # 
        addr = "192.1.10.200" #"192.1.10.162" #   "13.229.184.70" #
        port = 104 # 104 # 11114 #
        command = f"python SCU.py -a {ae_title_scp} -s {addr} -p {port} -f {dcm_compressed_path}"
        subprocess.run(command.split())

        print(f'Send {Accession_Number} Modified DICOM "{k}" with execution time: {time.time() - start_time :.2f} seconds')
        print('  {k} Done  '.center(100,'='))

    # # Load log_send_c_store single record to disk
    # folder_path = f'{BASE_DIR}/resources/log'
    # Path(folder_path).mkdir(parents=True, exist_ok=True)
    # path_store_log_send_c_store_single  = os.path.join(folder_path, ds.AccessionNumber +'.csv' )
    # metadata_df_SCU = pd.read_csv(path_store_log_send_c_store_single, error_bad_lines=False)

    # # Concat log_receive_dcm with log_send_c_store then save to local
    # folder_path = f'{BASE_DIR}/resources/log/log_send_c_store/{year}/{month}'
    # Path(folder_path).mkdir(parents=True, exist_ok=True)
    # path_store_log_concat = os.path.join(folder_path, str(date)+'.csv' )
    # metadata_df_SCU = metadata_df_SCU.drop(['ID', 'Accession Number', 'Study Date','Study Time'], axis='columns')
    # if not os.path.exists(path_store_log_concat):
    #     metadata_df_concat.to_csv(path_store_log_concat, index=False)
    # else:
    #     metadata_df_concat.to_csv(path_store_log_concat, index=False, mode='a', header=False)


    # deleting and clear the variable from memory in python
    del ds_modify
    # Delete temp file

    del ds, event
    gc.collect()
    with open(os.path.join(bbox_heatmap_dir, "success.txt"), 'w') as f:
        f.write('Save to PACS successfully')
    
    print('  Done  '.center(100,'='))

"""
LOCAL DIRECTORY
"""
def get_all_local(hn):
    try:
        all_data = []
        for file in os.listdir(LOCAL_DIR):
            if file.endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(LOCAL_DIR, file))
                if ds.PatientID == hn:
                    data = dict()
                    data['Accession No'] = ds.AccessionNumber
                    data['Modality'] = ds.Modality
                    data['Patient ID'] = ds.PatientID
                    data['Patient Name'] = ds.PatientName.given_name + " " + ds.PatientName.family_name
                    try:
                        data['Procedure Code'] = ds[0x020,0x0010].value
                    except:
                        data["Procedure Code"] = ""
                    try:
                        data['Study Date Time'] = str(pd.to_datetime(ds.StudyDate)) # ds.StudyTime
                    except:
                        data["Study Date Time"] = ""
                    # data['Proc Description'] = 'Chest PA upright'
                    all_data.append(data)
        if all_data != []:
            with open(os.path.join(TEMP_DIR, "local_dicom_files_{}.json".format(hn)), 'w') as f:
                json.dump(all_data, f)
    except Exception as e:
        print(traceback.format_exc())

def get_info_local(hn):
    try:
        all_data = {}
        for file in os.listdir(LOCAL_DIR):
            if file.endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(LOCAL_DIR, file))
                if ds.PatientID == hn:
                    data = dict()
                    data['Patient ID'] = ds.PatientID
                    data['Patient Name'] =  ds.PatientName.given_name + " " + ds.PatientName.family_name
                    data['Patient Sex'] = ds.PatientSex
                    try:
                        data['Patient Birthdate'] = str(pd.to_datetime(ds.PatientBirthDate))
                    except:
                        data['Patient Birthdate'] = ds.PatientBirthDate
                    if 'PatientAge' in ds:
                        data['Age'] = ds.PatientAge
                    all_data = data
                    with open(os.path.join(TEMP_DIR, "local_dicom_info_{}.json".format(hn)), 'w') as f:
                        json.dump(all_data, f)
                    break
    except Exception as e:
        print(traceback.format_exc())

def infer_local(acc_no, model_name):
    try:
        acc_no = str(acc_no)
        file_dir = os.path.join(TEMP_DIR, "local_{}_{}".format(model_name, acc_no))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        for file in os.listdir(LOCAL_DIR):
            if file.endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(LOCAL_DIR, file))
                if ds.AccessionNumber == acc_no:
                    message = "Error occurred"
                    result = False
                    if "classification_pylon" in model_name:
                        result, message = pylon_predict(ds, file_dir, model_name)
                    # elif model_name == "covid19_admission":
                    #     result = covid_predict(file_dir)
                    else:
                        message = "Model not found"

                    if result:
                        with open(os.path.join(file_dir, "success.txt"), 'w') as f:
                            f.write('infer successfully')
                    else:
                        with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
                            f.write(message)
        else:
            with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
                f.write('Cannot find dicom file')
    except Exception as e:
        print(traceback.format_exc())

def main() -> None:
    hn = args.hn
    func = args.function
    acc_no = args.accession_no
    model = args.model
    bbox = args.bounding_box
    if func == "get_info":
        get_info(hn)
    elif func == "get_all":
        get_all(hn)
    elif func == "get_dicom":
        get_dicom(acc_no)
    elif func == "infer":
        infer(acc_no, model)
    elif func == "save_to_pacs":
        save_to_pacs(acc_no, bbox)
    elif func == "get_info_local":
        get_info_local(hn)
    elif func == "get_all_local":
        get_all_local(hn)
    elif func == "infer_local":
        infer_local(acc_no, model)

if __name__ == "__main__":
    main()


# def plot_bbox_from_df(df_bbox, image_path):
#     image_path = os.path.join(PACS_DIR, image_path)
#     ### df_bbox คือ json ที่ได้มาจาก database
#     ### image_path คือ path ของ image

#     import cv2
#     import pydicom
#     import numpy as np
#     import pandas as pd

#     all_bbox = pd.DataFrame(df_bbox['data'], columns=['label', 'tool', 'data'])

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 1.5
#     thickness = 6
#     isClosed = True
#     # Green color
#     color = (0, 255, 0)

#     ### ส่วนที่เรียก image
#     #import dicom image array
#     if image_path != None:
#         event = load_file(image_path)

#         dicom = event.dataset
#         dicom.file_meta = event.file_meta

#         # dicom = pydicom.dcmread(image_path)
#         inputImage = dicom.pixel_array

#         # depending on this value, X-ray may look inverted - fix that:
#         if dicom.PhotometricInterpretation == "MONOCHROME1":
#             inputImage = np.amax(inputImage) - inputImage
#             inputImage = np.stack([inputImage, inputImage, inputImage])
#             inputImage = inputImage.astype('float32')
#             inputImage = inputImage - inputImage.min()
#             inputImage = inputImage / inputImage.max()
#             inputImage = inputImage.transpose(1, 2, 0)
#             inputImage = (inputImage*255).astype(int)
#             # https://github.com/opencv/opencv/issues/14866
#             inputImage = cv2.UMat(inputImage).get()
#     else:
#         inputImage = np.zeros([3000, 3000, 3])
#         inputImage = inputImage.astype(int)
#         inputImage = cv2.UMat(inputImage).get()

#     for index, row in all_bbox.iterrows():

#         if row['tool'] == 'rectangleRoi':
#             pts = row["data"]["handles"]
#             inputImage = cv2.rectangle(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
#                 pts["end"]["x"]), int(pts["end"]["y"])), color, thickness)
#             inputImage = cv2.putText(inputImage,  row["label"], (int(min(pts["start"]["x"], pts["end"]["x"])), int(
#                 min(pts["start"]["y"], pts["end"]["y"]))), font, fontScale, color, thickness, cv2.LINE_AA)

#         if row['tool'] == 'freehand':
#             pts = np.array([[cdn["x"], cdn["y"]]
#                            for cdn in row["data"]["handles"]], np.int32)
#             pts = pts.reshape((-1, 1, 2))
#             #choose min x,y for text origin
#             text_org = np.amin(pts, axis=0)
#             inputImage = cv2.polylines(inputImage, [pts], isClosed, color, thickness)
#             inputImage = cv2.putText(inputImage,  row["label"], tuple(
#                 text_org[0]), font, fontScale, color, thickness, cv2.LINE_AA)

#         if row['tool'] == 'length':
#             pts = row["data"]["handles"]
#             #choose left point for text origin
#             text_org = "start"
#             if pts["start"]["x"] > pts["end"]["x"]:
#                 text_org = "end"
#             inputImage = cv2.line(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
#                 pts["end"]["x"]), int(pts["end"]["y"])), color, thickness)
#             inputImage = cv2.putText(inputImage,  row["label"], (int(pts[text_org]["x"]), int(
#                 pts[text_org]["y"])), font, fontScale, color, thickness, cv2.LINE_AA)

#         if row['tool'] == 'ratio':
#             pts = row["data"]["0"]["handles"]
#             #choose left point for text origin
#             text_org = "start"
#             if pts["start"]["x"] > pts["end"]["x"]:
#                 text_org = "end"
#             inputImage = cv2.line(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
#                 pts["end"]["x"]), int(pts["end"]["y"])), color, thickness)
#             inputImage = cv2.putText(inputImage,  row["label"], (int(pts[text_org]["x"]), int(
#                 pts[text_org]["y"])), font, fontScale, color, thickness, cv2.LINE_AA)
#             pts = row["data"]["1"]["handles"]
#             #choose left point for text origin
#             text_org = "start"
#             if pts["start"]["x"] > pts["end"]["x"]:
#                 text_org = "end"
#             inputImage = cv2.line(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
#                 pts["end"]["x"]), int(pts["end"]["y"])), color, thickness)
#             inputImage = cv2.putText(inputImage,  row["label"], (int(pts[text_org]["x"]), int(
#                 pts[text_org]["y"])), font, fontScale, color, thickness, cv2.LINE_AA)

#         #cv2.imshow(inputImage)
#         cv2.imwrite(os.path.join(PACS_DIR, "save.png"), inputImage)
#     return inputImage