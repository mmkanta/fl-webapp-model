from starlette.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTasks
from fastapi import APIRouter, Form, File, UploadFile, Body
import shutil
import subprocess
import traceback
import time

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

@router.get("/HN/{hn}", status_code=200)
async def get_all_by_hn(hn: str):
    try: 
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-n", hn, "-f", "get_all_local"])
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
        file_location = os.path.join(LOCAL_DIR, "{}.dcm".format(acc_no))

        if os.path.exists(os.path.join(file_location)):
            return FileResponse(file_location)
        else:
            return JSONResponse(content={"success": False, "message": "Cannot find dicom file"}, status_code=400)
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
            return JSONResponse(content={"success": False, "message": message}, status_code=500)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)