from starlette.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTasks
from fastapi import APIRouter
from pacs_connection.Utilis_DICOM import check_dicom_exist, get_dicom_location
import shutil
import subprocess
import traceback
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

import os
import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(BASE_DIR, 'resources', 'local')
TEMP_DIR = os.path.join(BASE_DIR, 'resources', 'temp')

class Data(BaseModel):
    data: List[dict]

class BBox(BaseModel):
    bbox_data: str

def remove_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print('Finish clearing temporary file')

# get all dicom's info by condition
@router.get("/HN/", status_code=200)
async def get_all(hn: Optional[str] = None, acc_no: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
    try: 
        if hn == None: hn = "None"
        if acc_no == None: acc_no = "None"
        if start_date == None: start_date = "None"
        if end_date == None: end_date = "None"
        
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
        # print(e)
        return JSONResponse(content={"success": False, "message": e}, status_code=500)

# get patient's info
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
        # print(e)
        return JSONResponse(content={"success": False, "message": e}, status_code=500)

# get dicom file
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
        # print(e)
        return JSONResponse(content={"success": False, "message": e}, status_code=500)

# inference
@router.get("/infer/{model_name}/{acc_no}", status_code=200)
async def infer(model_name: str, acc_no: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    now = datetime.now().strftime("%H%M%S")

    file_dir = os.path.join(TEMP_DIR, "local_{}_{}_{}".format(model_name, acc_no, now))

    # remove directory after finish process
    background_tasks.add_task(remove_file, file_dir)
    try:
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "infer_local", "-a", acc_no, "-m", model_name, "-s", now])
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
        # print(e)
        return JSONResponse(content={"success": False, "message": e}, status_code=500)

# get bounding box image
@router.post("/png", status_code=200)
async def generate_png_bbox(bbox: BBox, background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        bbox_data = bbox.bbox_data
        bbox_dict = json.loads(bbox_data)
        acc_no = bbox_dict["accession_no"]

        bbox_dir = os.path.join(BASE_DIR, "resources", "temp", acc_no + "_bbox")
        if not os.path.exists(bbox_dir):
            os.makedirs(bbox_dir)

        background_tasks.add_task(remove_file, bbox_dir)

        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "get_bbox_img_local", "-a", acc_no, "-b", bbox_data])

        if os.path.exists(os.path.join(bbox_dir, "rendered_bbox_image.png")):
            return FileResponse(os.path.join(bbox_dir, "rendered_bbox_image.png"))

        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)
    except Exception as e:
        print(traceback.format_exc())
        # print(e)
        return JSONResponse(content={"success": False, "message": e}, status_code=500)

# get filename and directory of dicom file by accession number
@router.patch("/loc/", status_code=200)
async def get_location(req_body: Data):
    try:
        new_data = get_dicom_location(req_body.data)
        return JSONResponse(content={"success": True, "message": "Get dicom location successfully", "data": new_data}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        # print(e)
        return JSONResponse(content={"success": False, "message": e}, status_code=500)