import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traceback
import numpy as np
import pandas as pd
import pickle
import subprocess
import pydicom
from pathlib import Path
import shutil
import gc
import argparse
import json
from evt_classes import *
from Utilis_DICOM import array_to_dicom, plot_bbox_from_df, get_png_file, get_all_ds_info, extract_date
from Constant import PACS_ADDR, PACS_PORT, AE_TITLE_SCP
from model.classification_pylon.predict import main as pylon_predict
from model.covid19_admission.predict_admission import main as covid_predict
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
parser.add_argument('-d', '--data', type=str, default="", help="Related Data")

args = parser.parse_args()

# dicom function called by /api

def save_file(file, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(file, f)

def load_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# get all dicom's info by condition
def get_all(hn, acc_no, start_date, end_date, acc_no_list, now):
    try:
        if hn == "None": hn = None
        if acc_no == "None": acc_no = None
        if start_date == "None": start_date = None
        if end_date == "None": end_date = None
        if acc_no_list == "None": acc_no_list = None
        try:
            acc_no_list = acc_no_list.split(' ')
            print(acc_no_list)
            all_data = []
            for acc_no in acc_no_list:
                try:
                    event = load_file(os.path.join(PACS_DIR, acc_no + '.evt'))
                    ds = event.dataset
                    data = get_all_ds_info(ds)
                    all_data.append(data)
                except:
                    continue
            with open(os.path.join(TEMP_DIR, "dicom_files_{}.json".format(now)), 'w') as f:
                json.dump(all_data, f)
            return
        except:
            pass

        all_data = []
        for file in os.listdir(PACS_DIR):
            if file.endswith('.evt'):
                event = load_file(os.path.join(PACS_DIR, file))
                ds = event.dataset
                dcm_date = extract_date(ds.StudyDate, ds.StudyTime)
                if ds.PatientID == 'anonymous':
                    continue
                if (not hn) and (not acc_no) and (not start_date) and (not end_date):
                    data = get_all_ds_info(ds)
                    all_data.append(data)
                elif ((not hn) or (hn and (hn == ds.PatientID))) \
                    and ((not acc_no) or (acc_no and (acc_no == ds.AccessionNumber))) \
                    and ((not start_date) or (start_date and (dcm_date >= datetime.datetime.fromtimestamp(int(start_date)/1000)))) \
                    and ((not end_date) or (end_date and (dcm_date <= datetime.datetime.fromtimestamp(int(end_date)/1000)))):
                    data = get_all_ds_info(ds)
                    all_data.append(data)
        with open(os.path.join(TEMP_DIR, "dicom_files_{}.json".format(now)), 'w') as f:
            json.dump(all_data, f)
    except Exception as e:
        print(traceback.format_exc())

# get patient's info
def get_info(hn):
    try:
        all_data = {}
        for file in os.listdir(PACS_DIR):
            if file.endswith('.evt'):
                event = load_file(os.path.join(PACS_DIR, file))
                ds = event.dataset
                if ds.PatientID == 'anonymous':
                    continue
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

# get dicom file
def get_dicom(acc_no):
    file_dir = os.path.join(TEMP_DIR, "{}_get".format(acc_no))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    try:
        acc_no = str(acc_no)
        
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
        with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
            f.write(str(e))

# inference
def infer(acc_no, model_name, start_time, data):
    file_dir = os.path.join(TEMP_DIR, "{}_{}_{}".format(model_name, acc_no, start_time))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    try:
        try:
            record = json.loads(data)
        except:
            record = {}

        acc_no = str(acc_no)
        
        res_dir = os.path.join(file_dir, 'result')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        if os.path.exists(os.path.join(PACS_DIR, acc_no + '.evt')):
            event = load_file(os.path.join(PACS_DIR, acc_no + '.evt'))
            ds = event.dataset
            ds.file_meta = event.file_meta

            if ds.PatientID == 'anonymous':
                with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
                    f.write('Cannot infer this dicom')
                return
            
            # save original file in png format
            status, message = get_png_file(ds, res_dir)
            if not status:
                with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
                    f.write(message)
                return

            # for model
            # result must be saved in prediction.txt in res_dir
            # if there is gradcam image, gradcam image also must be saved in res_dir
            message = "Error occurred"
            result = False
            if "classification_pylon" in model_name:
                result, message = pylon_predict(ds, res_dir, model_name)
            elif model_name == "covid19_admission":
                result, message = covid_predict(ds, res_dir, record)
            else:
                message = "Model not found"
            if result:
                with open(os.path.join(file_dir, "success.txt"), 'w') as f:
                    f.write('infer successfully')
            else:
                with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
                    f.write(str(message))
        else:
            with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
                f.write('Cannot find dicom file')
    except Exception as e:
        print(traceback.format_exc())
        with open(os.path.join(file_dir, "fail.txt"), 'w') as f:
            f.write(str(e))

# save dicom to PACS
def save_to_pacs(acc_no, bbox, model):
    try:
        # bbox string to dict
        bbox_dict = json.loads(bbox)
        bbox_heatmap_dir = os.path.join(BASE_DIR, 'resources', 'temp', acc_no + '_store')

        if not os.path.exists(os.path.join(PACS_DIR, acc_no + '.evt')):
            with open(os.path.join(bbox_heatmap_dir, "fail.txt"), 'w') as f:
                f.write('File not found')
            return
        
        event = load_file(os.path.join(PACS_DIR, acc_no + '.evt'))

        ds = event.dataset
        ds.file_meta = event.file_meta

        if ds.PatientID == 'anonymous':
            with open(os.path.join(bbox_heatmap_dir, "fail.txt"), 'w') as f:
                f.write('Cannot save this dicom')
            return

        Accession_Number = ds.AccessionNumber

        SCU_path = os.path.join(BASE_DIR, "pacs_connection", "SCU.py")
        # save all heatmap images to PACS
        for file in os.listdir(bbox_heatmap_dir):
            if file.endswith('.png'):
                filename = os.fsdecode(file)
                finding = filename.split('.')[0]
                ds_modify, dcm_compressed_path = array_to_dicom(ds, bbox_heatmap_dir, filename, model)
                if dcm_compressed_path is None:
                    print(f'Cannot convert image {finding} to dicom')
                    with open(os.path.join(bbox_heatmap_dir, "fail.txt"), 'w') as f:
                        f.write(f'Cannot convert image {finding} to dicom')
                    return

                # print(f'Receive DICOM and processing complete with execution time: {time.time() - start_time :.2f} seconds')  

                # SCU Role
                start_time = time.time()
    
                command = f"python {SCU_path} -a {AE_TITLE_SCP} -s {PACS_ADDR} -p {PACS_PORT} -f {dcm_compressed_path} -t {finding}"
                subprocess.run(command.split())

                del ds_modify

                print(f'Send {Accession_Number} Modified DICOM "{finding}" with execution time: {time.time() - start_time :.2f} seconds')
                print(f'  {finding} Done  '.center(100,'='))
                
        if isinstance(bbox_dict['data'], list) and len(bbox_dict['data']) > 0:
            # save rendered image to PACS
            plot_bbox_from_df(bbox_dict, ds, os.path.join(bbox_heatmap_dir, 'rendered_bbox_image.png'))
            ds_modify, dcm_compressed_path = array_to_dicom(ds, bbox_heatmap_dir, 'rendered_bbox_image.png', model)

            # SCU Role
            start_time = time.time()
            finding_type = 'Rendered_bounding_box'
            command = f"python {SCU_path} -a {AE_TITLE_SCP} -s {PACS_ADDR} -p {PACS_PORT} -f {dcm_compressed_path}  -t {finding_type}"
            subprocess.run(command.split())

            del ds_modify

            print(f'Send {Accession_Number} Modified DICOM "Rendered_bbox_image" with execution time: {time.time() - start_time :.2f} seconds')
            print('  Rendered Bounding Box Done  '.center(100,'='))

        if not os.path.exists(os.path.join(bbox_heatmap_dir, "fail.txt")):
            event.dataset.PatientID = 'anonymous'
            event.dataset.PatientName = '-^-'
            fake_event = FakeEvent(event)

            save_file(fake_event, os.path.join(PACS_DIR, acc_no + '.evt'))
            del fake_event
            print('  Create Dummy Dicom Done  '.center(100,'='))

        # deleting and clear the variable from memory in python
        del ds, event

        gc.collect()
        
        print('  Done  '.center(100,'='))
    except Exception as e:
        bbox_heatmap_dir = os.path.join(BASE_DIR, 'resources', 'temp', ds.AccessionNumber + '_store')
        with open(os.path.join(bbox_heatmap_dir, "fail.txt"), 'w') as f:
            f.write(str(e))
        print(e)

def convert_evt_to_dummy(acc_no_list):
    acc_no_list = acc_no_list.split(' ')
    today_date = datetime.date.today()
    dummy_day = today_date- datetime.timedelta(days=30)
    year = str(dummy_day.year)
    month = str(dummy_day.month)

    folder_path = os.path.join(BASE_DIR, 'resources', 'log', 'log_clear_dicom', 'dummy_dicom', year, month)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    log_clear_dicom_path = os.path.join(folder_path, str(today_date)+'.csv')
    for acc_no in acc_no_list:
        log_data = {'Accession Number': acc_no, "Success": ""}

        if os.path.exists(os.path.join(PACS_DIR, acc_no + '.evt')):
            try:
                event = load_file(os.path.join(PACS_DIR, acc_no + '.evt'))

                if event.dataset.PatientID == 'anonymous':
                    continue
                event.dataset.PatientID = 'anonymous'
                event.dataset.PatientName = '-^-'
                fake_event = FakeEvent(event)

                save_file(fake_event, os.path.join(PACS_DIR, acc_no + '.evt'))
                
                log_data['Success'] = True

            except Exception as e:
                log_data['Success'] = False
                print(e)
        log_data = pd.DataFrame.from_records([log_data])
        if not os.path.exists(log_clear_dicom_path):
            log_data.to_csv(log_clear_dicom_path, index=False)
        else:
            log_data.to_csv(log_clear_dicom_path, index=False, mode='a', header=False)

def main() -> None:
    hn = args.hn
    func = args.function
    acc_no = args.accession_no
    acc_no_list = args.accession_no_list
    model = args.model
    bbox = args.bounding_box
    start_date = args.start_date
    end_date = args.end_date
    data = args.data
    if func == "get_info":
        get_info(hn)
    elif func == "get_all":
        get_all(hn, acc_no, start_date, end_date, acc_no_list, data)
    elif func == "get_dicom":
        get_dicom(acc_no)
    elif func == "infer":
        infer(acc_no, model, start_date, data)
    elif func == "save_to_pacs":
        save_to_pacs(acc_no, bbox, model)
    elif func == "convert_evt_to_dummy":
        convert_evt_to_dummy(acc_no_list)

if __name__ == "__main__":
    main()