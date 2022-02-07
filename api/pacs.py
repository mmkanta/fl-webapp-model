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
import pandas as pd
import datetime
from zipfile import ZipFile
from Constant import MONGO_URL, SECRET

import os, sys
import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACS_DIR = os.path.join(BASE_DIR, "resources", "files")
TEMP_DIR = os.path.join(BASE_DIR, "resources", "temp")

# validate user when save dicom to PACS
def validate_authorization(authorization, result_id):
    token = ""
    try:
        token = authorization.split(" ")[1]
        jwt.decode(token, SECRET, algorithms=["HS256"])
    except:
        print(traceback.format_exc())
        return False, "Token is invalid"
    try:
        db = pymongo.MongoClient(MONGO_URL)["webapp"]
        user = db["users"].find_one({"token": token})
        result = db["pred_results"].find_one({"_id": ObjectId(result_id)})
        project = db["projects"].find_one({"_id": result["project_id"]})
        if user["_id"] in project["head"]:
            return True, ""
        return False, ""
    except:
        print(traceback.format_exc())
        return False, "User is unauthorized or cannot connect to database"

def remove_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Finish clearing temporary file")

# get all dicom's info by condition
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
            "-n", hn, "-f", "get_all", "-a", acc_no, "-s", start_date, "-e", end_date
        ])
        if os.path.exists(os.path.join(TEMP_DIR, "dicom_files_{}.json".format(hn))):
            with open(os.path.join(TEMP_DIR, "dicom_files_{}.json".format(hn)), "r") as f:
                data = json.load(f)
            os.remove(os.path.join(TEMP_DIR, "dicom_files_{}.json".format(hn)))
        else:
            data = {}
        return JSONResponse(content={"success": True, "message": "Get dicom files by HN successfully", "data": data}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        # print(e)
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

# get patient's info
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
        # print(e)
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

# get dicom file
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
        # print(e)
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

# save dicom to PACS
@router.post("/save", status_code=200)
async def save_to_pacs(authorization: Optional[str] = Header(None), bbox_data: str = Form(...), file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        print("Start save to PACS process")
        # string to dict
        bbox_dict = json.loads(bbox_data)
        success, message = validate_authorization(authorization, bbox_dict["result_id"])
        if not success:
            print(message)
            return JSONResponse(content={"success": False, "message": message}, status_code=400)

        acc_no = bbox_dict["acc_no"]

        bbox_heatmap_dir = os.path.join(BASE_DIR, "resources", "temp", acc_no + "_store")
        if not os.path.exists(bbox_heatmap_dir):
            os.makedirs(bbox_heatmap_dir)

        background_tasks.add_task(remove_file, bbox_heatmap_dir)

        with open(os.path.join(bbox_heatmap_dir, file.filename), "wb") as f:
            f.write(file.file.read())

        # extract zip file of heatmap
        with ZipFile(os.path.join(bbox_heatmap_dir, file.filename), 'r') as zip:
            zip.extractall(path=bbox_heatmap_dir)

        # delete zip file
        os.remove(os.path.join(bbox_heatmap_dir, file.filename))

        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "save_to_pacs", "-a", acc_no, "-b", bbox_data])

        if os.path.exists(os.path.join(bbox_heatmap_dir, "fail.txt")):
            message = "error"
            with open(os.path.join(bbox_heatmap_dir, "fail.txt"), "r") as f:
                message = f.readline()
            print(message)
            return JSONResponse(content={"success": False, "message": message}, status_code=500)

        return JSONResponse(content={"success": True, "message": "Save DICOM to PACS successfully"}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        # print(e)
        return JSONResponse(content={"success": False, "message": e}, status_code=500)

@router.delete("/clear", status_code=200)
async def clear_dicom_folder():
    LOG_DIR = os.path.join(BASE_DIR, "resources", "log", "log_receive_dcm")
    today_date = datetime.date.today()
    try:
        delete_date = today_date
        n = today_date.day
        if today_date.day == 28: # delete dicom received before 14th
            delete_date = today_date.replace(day=14)
            n = 28
        elif today_date.day == 14: # delete dicom received before 28th
            delete_date = today_date.replace(day=28, month=today_date.month-1)
            n = 14
        while delete_date.day != n:
            year = delete_date.year
            month = delete_date.month
            df = pd.read_csv(os.path.join(LOG_DIR, year, month, str(delete_date) + '.csv'))
            unique_acc_no = set(df["Accession Number"])
            # check acc no in mongo
            db = pymongo.MongoClient(MONGO_URL)["webapp"]
            used_acc_no = db["images"].distinct("accession_no", {"accession_no": {"$in": list(unique_acc_no)}})
            for acc_no in unique_acc_no:
                if acc_no not in used_acc_no and os.path.exists(os.path.join(PACS_DIR, acc_no + '.evt')):
                    os.remove(os.path.join(PACS_DIR, acc_no + '.evt'))
            delete_date = delete_date - datetime.timedelta(days=1)
        
        if today_date.day == 28: # change original dicom to dummy dicom
            delete_date = today_date.replace(day=28, month=today_date.month-1)
            while True:
                year = delete_date.year
                month = delete_date.month
                df = pd.read_csv(os.path.join(LOG_DIR, year, month, str(delete_date) + '.csv'))
                unique_acc_no = set(df["Accession Number"])
                used_acc_no = db["images"].distinct("accession_no", {
                    "accession_no": {"$in": list(unique_acc_no)},
                    "hn": {"$ne": ""}
                    })
                subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "get_dummy", "-l", used_acc_no])
                delete_date = delete_date - datetime.timedelta(days=1)

                image_id_list = db["images"].find({"accession_no": {"$in": used_acc_no}}, {"_id"})
                result_list = db["pred_results"].find({"image_id", {"$in": image_id_list}})
                for result in result_list:
                    db["pred_results"].find_one_and_update({"_id": result["_id"]}, {
                        "hn": None,
                        "patient_name": None
                    })
                    db["images"].find_one_and_update({"_id": result["image_id"]}, {
                        "hn": None,
                    })
                    db["medrecords"].find_one_and_update({"_id": result["record_id"]}, {
                        "record.hn": None,
                    })
                if delete_date.day == 28:
                    break
        del df
        return JSONResponse(content={"success": True, "message": "Finish clearing dicom"}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "message": e}, status_code=500)