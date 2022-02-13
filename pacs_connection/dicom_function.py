import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traceback
# import numpy as np
import pandas as pd
# import pickle
# import subprocess
import pydicom
from datetime import datetime
# from pathlib import Path
# import shutil
# import gc
import argparse
import json
# from evt_classes import *
from Utilis_DICOM import array_to_dicom, plot_bbox_from_df
# from Constant import PACS_ADDR, PACS_PORT, AE_TITLE_SCP
from model.classification_pylon.predict import main as pylon_predict
import datetime

# from model.covid19_admission.predict_admission import main as covid_predict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACS_DIR = os.path.join(BASE_DIR, 'resources', 'files')
TEMP_DIR = os.path.join(BASE_DIR, 'resources', 'temp')
LOCAL_DIR = os.path.join(BASE_DIR, 'resources', 'local')
sys.path.append(BASE_DIR)

# https://zetcode.com/python/argparse/
parser = argparse.ArgumentParser()
   
parser.add_argument('-n', '--hn', type=str, default='', help="Patient's HN")
parser.add_argument('-f', '--function', type=str, default="", help="Function's Name")
parser.add_argument('-a', '--accession_no', type=str, default="", help="Accession Number")
parser.add_argument('-l', '--accession_no_list', type=str, default="", help="Accession Number List")
parser.add_argument('-m', '--model', type=str, default="", help="Model's Name")
parser.add_argument('-b', '--bounding_box', type=str, default="", help="Bounding Box Dict")
parser.add_argument('-s', '--start_date', type=str, default="", help="Start Date")
parser.add_argument('-e', '--end_date', type=str, default="", help="End Date")

args = parser.parse_args()

# dicom function called by /api

# def load_file(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data

# extract dicom info for 'select cxr' page
def extract_ds_info(ds):
    data = dict()
    data['Accession No'] = ds.AccessionNumber
    data['Modality'] = ds.Modality
    data['Patient ID'] = int(ds.PatientID)
    data['Patient Name'] = ds.PatientName.given_name + " " + ds.PatientName.family_name
    data['Patient Sex'] = ds.PatientSex
    try:
        data['Age'] = int(ds.PatientAge.split('Y')[0])
    except:
        data['Age'] = None
    try:
        data['Procedure Code'] = ds[0x020,0x0010].value
    except:
        data["Procedure Code"] = ""
    try:
        data['Study Date Time'] = str(pd.to_datetime(ds.StudyDate)) # ds.StudyTime
    except:
        data["Study Date Time"] = ""
    # data['Proc Description'] = 'Chest PA upright'
    return data

"""
LOCAL DIRECTORY
"""
def get_all_local(hn, acc_no, start_date, end_date):
    try:
        if hn == "None": hn = None
        if acc_no == "None": acc_no = None
        if start_date == "None": start_date = None
        if end_date == "None": end_date = None

        all_data = []
        for file in os.listdir(LOCAL_DIR):
            if file.endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(LOCAL_DIR, file))
                if (not hn) and (not acc_no) and (not start_date) and (not end_date):
                    data = extract_ds_info(ds)
                    all_data.append(data)
                elif ((not hn) or (hn and (hn in ds.PatientID))) \
                    and ((not acc_no) or (acc_no and (acc_no in ds.AccessionNumber))) \
                    and ((not start_date) or (start_date and (pd.to_datetime(ds.StudyDate, infer_datetime_format=True) >= datetime.datetime.fromtimestamp(int(start_date)/1000)))) \
                    and ((not end_date) or (end_date and (pd.to_datetime(ds.StudyDate, infer_datetime_format=True) <= datetime.datetime.fromtimestamp(int(end_date)/1000)))):
                    data = extract_ds_info(ds)
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
                    break
        else:
            with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
                f.write('Cannot find dicom file')
    except Exception as e:
        print(traceback.format_exc())

def get_bbox_img_local(acc_no, bbox):
    bbox_dict = json.loads(bbox)
    bbox_dir = os.path.join(TEMP_DIR, acc_no + '_bbox')
    if isinstance(bbox_dict['data'], list) and len(bbox_dict['data']) > 0:
        for file in os.listdir(LOCAL_DIR):
            if file.endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(LOCAL_DIR, file))
                if ds.AccessionNumber == acc_no:
                    plot_bbox_from_df(bbox_dict, ds, os.path.join(bbox_dir, 'rendered_bbox_image.png'))  
                    break
    return

def main() -> None:
    hn = args.hn
    func = args.function
    acc_no = args.accession_no
    acc_no_list = args.accession_no_list
    model = args.model
    bbox = args.bounding_box
    start_date = args.start_date
    end_date = args.end_date
    # if func == "get_info":
    #     get_info(hn)
    # elif func == "get_all":
    #     get_all(hn, acc_no, start_date, end_date)
    # elif func == "get_dicom":
    #     get_dicom(acc_no)
    # elif func == "infer":
    #     infer(acc_no, model)
    # elif func == "save_to_pacs":
    #     save_to_pacs(acc_no, bbox)
    if func == "get_info_local":
        get_info_local(hn)
    elif func == "get_all_local":
        get_all_local(hn, acc_no, start_date, end_date)
    elif func == "infer_local":
        infer_local(acc_no, model)
    elif func == "get_bbox_img_local":
        get_bbox_img_local(acc_no, bbox)

if __name__ == "__main__":
    main()