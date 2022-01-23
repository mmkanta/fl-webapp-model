from starlette.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTasks
from fastapi import APIRouter #, Form, File, UploadFile, Body
from pacs_connection.Utilis_DICOM import check_dicom_exist
import shutil
import subprocess
import traceback
import time
from typing import Optional
import datetime
from dateutil import parser

import os, sys
import json

router = APIRouter()

async def remove_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print('Finish clearing temporary file')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(BASE_DIR, 'resources', 'local')
TEMP_DIR = os.path.join(BASE_DIR, 'resources', 'temp')

@router.get("/HN/", status_code=200)
async def get_all(hn: Optional[str] = None, acc_no: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
    try: 
        if hn == None: hn = "None"
        if acc_no == None: acc_no = "None"
        if start_date == None: start_date = "None"
        if end_date == None: end_date = "None"
        # print(datetime.datetime.fromtimestamp(int(start_date)/1000))
        subprocess.run([
            "python", 
            os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), 
            "-n", hn, "-f", "get_all_local", "-a", acc_no, "-s", start_date, "-e", end_date
        ])
        if os.path.exists(os.path.join(TEMP_DIR, "local_dicom_files_{}.json".format(hn))):
            with open(os.path.join(TEMP_DIR, "local_dicom_files_{}.json".format(hn)), "r") as f:
                data = json.load(f)
            os.remove(os.path.join(TEMP_DIR, "local_dicom_files_{}.json".format(hn)))
        else:
            data = {}
        return JSONResponse(content={"success": True, "message": "Get dicom files by HN successfully", "data": data}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

@router.get("/HN/{hn}/info", status_code=200)
async def get_info_by_hn(hn: str):
    try: 
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-n", hn, "-f", "get_info_local"])
        if os.path.exists(os.path.join(TEMP_DIR, "local_dicom_info_{}.json".format(hn))):
            with open(os.path.join(TEMP_DIR, "local_dicom_info_{}.json".format(hn)), "r") as f:
                data = json.load(f)
            os.remove(os.path.join(TEMP_DIR, "local_dicom_info_{}.json".format(hn)))
        else:
            data = {}
        return JSONResponse(content={"success": True, "message": "Get patient's info successfully", "data": data}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

@router.get("/acc_no/{acc_no}", status_code=200)
async def get_dicom_file(acc_no: str):
    try:
        exist, filename = check_dicom_exist(acc_no)
        if not exist:
            return JSONResponse(content={"success": False, "message": "Cannot find dicom file"}, status_code=400)
        file_location = os.path.join(LOCAL_DIR, filename)

        return FileResponse(file_location)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

@router.get("/infer/{model_name}/{acc_no}", status_code=200)
async def infer(model_name: str, acc_no: str, background_tasks: BackgroundTasks = BackgroundTasks()):

    file_dir = os.path.join(TEMP_DIR, "local_{}_{}".format(model_name, acc_no))

    # remove the directory
    background_tasks.add_task(remove_file, file_dir)
    try:
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "infer_local", "-a", acc_no, "-m", model_name])
        if os.path.exists(os.path.join(file_dir, 'success.txt')):
            shutil.make_archive(os.path.join(file_dir, acc_no),
                                'zip',
                                root_dir=os.path.join(file_dir, 'result'),
                                )
            print("Inference end")
            return FileResponse(os.path.join(file_dir, acc_no + '.zip'))
        elif os.path.exists(os.path.join(file_dir, 'fail.txt')):
            message = "error"
            with open(os.path.join(file_dir, 'fail.txt'), "r") as f:
                message = f.readline()
                print(message)
            return JSONResponse(content={"success": False, "message": message}, status_code=500)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)