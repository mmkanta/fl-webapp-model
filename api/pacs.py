from starlette.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTasks
from fastapi import APIRouter, Form, File, UploadFile
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
DICOM_DIR = os.path.join(BASE_DIR, 'resources', 'files')
TEMP_DIR = os.path.join(BASE_DIR, 'resources', 'temp')

@router.get("/HN/{hn}", status_code=200)
async def get_all_by_hn(hn: str):
    try: 
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-n", hn, "-f", "get_all"])
        if os.path.exists(os.path.join(TEMP_DIR, "dicom_files_{}.json".format(hn))):
            with open(os.path.join(TEMP_DIR, "dicom_files_{}.json".format(hn)), "r") as f:
                data = json.load(f)
            os.remove(os.path.join(TEMP_DIR, "dicom_files_{}.json".format(hn)))
        else:
            data = {}
        return JSONResponse(content={"success": True, "message": "Get dicom files by HN successfully", "data": data}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

@router.get("/HN/{hn}/info", status_code=200)
async def get_info_by_hn(hn: str):
    try: 
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-n", hn, "-f", "get_info"])
        if os.path.exists(os.path.join(TEMP_DIR, "dicom_info_{}.json".format(hn))):
            with open(os.path.join(TEMP_DIR, "dicom_info_{}.json".format(hn)), "r") as f:
                data = json.load(f)
            os.remove(os.path.join(TEMP_DIR, "dicom_info_{}.json".format(hn)))
        else:
            data = {}
        return JSONResponse(content={"success": True, "message": "Get patient's info successfully", "data": data}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

@router.get("/acc_no/{acc_no}", status_code=200)
async def get_dicom_file(acc_no: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        # gdcmconv -C save_test.dcm test.dcm
        file_dir = os.path.join(TEMP_DIR, "{}_get".format(acc_no))
        background_tasks.add_task(remove_file, os.path.join(file_dir))
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "get_dicom", "-a", acc_no])

        if os.path.exists(os.path.join(file_dir, "not_found.txt")):
            return JSONResponse(content={"success": False, "message": "Cannot find dicom file"}, status_code=400)

        if os.path.exists(os.path.join(file_dir, "{}.dcm".format(acc_no))):
            return FileResponse(os.path.join(file_dir, "{}.dcm".format(acc_no)))

        return JSONResponse(content={"success": False, "message": "Cannot find dicom file"}, status_code=400)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

@router.post("/save", status_code=200)
async def save_to_pacs(files: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        acc_no = files.filename.split('.')[0]
        bbox_heatmap_dir = os.path.join(BASE_DIR, 'resources', 'temp', acc_no + '_store')
        
        background_tasks.add_task(remove_file, bbox_heatmap_dir)

        with open(os.path.join(bbox_heatmap_dir + files.filename), "wb") as f:
            f.write(files.file.read())

        # unzip files

        os.remove(os.path.join(bbox_heatmap_dir + files.filename))

        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "save_to_pacs", "-a", acc_no])

        if os.path.exists(os.path.join(bbox_heatmap_dir, "fail.txt")):
            message = "error"
            with open(os.path.join(bbox_heatmap_dir, 'fail.txt'), "r") as f:
                message = f.readline()
            return JSONResponse(content={"success": False, "message": message}, status_code=500)

        if os.path.exists(os.path.join(bbox_heatmap_dir, "success.txt")):
            return JSONResponse(content={"success": True, "message": "Save DICOM to PACS successfully"}, status_code=200)

        return JSONResponse(content={"success": False, "message": "Cannot find dicom file"}, status_code=400)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

