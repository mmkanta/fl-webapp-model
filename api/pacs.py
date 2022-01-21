from starlette.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTasks
from fastapi import APIRouter, Form, File, UploadFile, Body, Request, Header
from typing import Optional
import shutil
import subprocess
import traceback
import time
import jwt
import pymongo
from bson.objectid import ObjectId
import json
from zipfile import ZipFile

import os, sys
import json

router = APIRouter()

SECRET = "oUQF9vv5MB77302BJm6HDKulKKPqfukuiW5zMeamAx2JJU21cJkx23MBShP3GVt"
CONNECTION_STRING = "mongodb://localhost/webapp"
# CONNECTION_STRING = "mongodb://admin:admin@mongo:27017/webapp?authSource=webapp&w=1"

def validate_authorization(authorization, result_id):
    token = ""
    try:
        token = authorization.split(" ")[1]
        jwt.decode(token, SECRET, algorithms=["HS256"])
    except:
        return False, "Token is invalid"
    try:
        db = pymongo.MongoClient(CONNECTION_STRING)["webapp"]
        user = db["users"].find_one({"token": token})
        result = db["pred_results"].find_one({"_id": ObjectId(result_id)})
        project = db["projects"].find_one({"_id": result["project_id"]})
        if user["_id"] in project["head"]:
            return True, ""
        return False, ""
    except:
        return False, "User is unauthorized or cannot connect to database"

def remove_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Finish clearing temporary file")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACS_DIR = os.path.join(BASE_DIR, "resources", "files")
TEMP_DIR = os.path.join(BASE_DIR, "resources", "temp")

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
async def save_to_pacs(authorization: Optional[str] = Header(None), bbox_data: str = Form(...), file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        bbox_dict = json.loads(bbox_data)
        success, message = validate_authorization(authorization, bbox_dict["result_id"])
        if not success:
            return JSONResponse(content={"success": False, "message": message}, status_code=400)

        acc_no = bbox_dict["acc_no"]

        bbox_heatmap_dir = os.path.join(BASE_DIR, "resources", "temp", acc_no + "_store")
        if not os.path.exists(bbox_heatmap_dir):
            os.makedirs(bbox_heatmap_dir)

        background_tasks.add_task(remove_file, bbox_heatmap_dir)

        # png file
        with open(os.path.join(bbox_heatmap_dir, file.filename), "wb") as f:
            f.write(file.file.read())

        with ZipFile(os.path.join(bbox_heatmap_dir, file.filename), 'r') as zip:
            zip.extractall(path=bbox_heatmap_dir)

        os.remove(os.path.join(bbox_heatmap_dir, file.filename))

        # dict to string
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "save_to_pacs", "-a", acc_no, "-b", bbox_data])

        if os.path.exists(os.path.join(bbox_heatmap_dir, "fail.txt")):
            message = "error"
            with open(os.path.join(bbox_heatmap_dir, "fail.txt"), "r") as f:
                message = f.readline()
            return JSONResponse(content={"success": False, "message": message}, status_code=500)

        if os.path.exists(os.path.join(bbox_heatmap_dir, "success.txt")):
            return JSONResponse(content={"success": True, "message": "Save DICOM to PACS successfully"}, status_code=200)

        return JSONResponse(content={"success": False, "message": "Cannot find dicom file"}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)
