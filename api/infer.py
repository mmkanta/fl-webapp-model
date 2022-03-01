from starlette.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTasks
from fastapi import APIRouter #, Form, File, UploadFile
# from model.classification_pylon.predict import main as pylon_predict
# from model.covid19_admission.predict_admission import main as covid_predict
import shutil
import subprocess
import time
import traceback
from datetime import datetime

import os
# import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, 'resources', 'temp')

def remove_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print('Finish clearing temporary file')

# inference
@router.get("/{model_name}/{acc_no}", status_code=200)
async def index(model_name: str, acc_no: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    now = datetime.now().strftime("%H%M%S%f")

    file_dir = os.path.join(TEMP_DIR, "{}_{}_{}".format(model_name, acc_no, now))

    # remove directory after finish process
    background_tasks.add_task(remove_file, file_dir)
    
    try:
        start_time = time.time()
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "infer", "-a", acc_no, "-m", model_name, "-s", now])
        
        if os.path.exists(os.path.join(file_dir, 'success.txt')):
            shutil.make_archive(os.path.join(file_dir, acc_no),
                                'zip',
                                root_dir=os.path.join(file_dir, 'result'),
                                )
            print(f"Inference end time: {time.time() - start_time :.2f} seconds")
            return FileResponse(os.path.join(file_dir, acc_no + '.zip'), status_code=200)
        elif os.path.exists(os.path.join(file_dir, 'fail.txt')):
            message = "error"
            with open(os.path.join(file_dir, 'fail.txt'), "r") as f:
                message = f.readline()
            return JSONResponse(content={"success": False, "message": message}, status_code=500)

    except Exception as e:
        print(traceback.format_exc())
        # print(e)
        return JSONResponse(content={"success": False, "message": e}, status_code=500)