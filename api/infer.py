from starlette.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTasks
from fastapi import APIRouter #, Form, File, UploadFile
# from model.classification_pylon.predict import main as pylon_predict
# from model.covid19_admission.predict_admission import main as covid_predict
import shutil
import subprocess
import time
import traceback

import os
# import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, 'resources', 'temp')

async def remove_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print('Finish clearing temporary file')

# @router.post("", status_code=200)
# async def index(model_name: str = Form(...), file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
#     path = os.path.join(BASE_DIR, "resources", "temp", file.filename.split('.')[0])
#     file_directory = path

#     count = 1
#     while os.path.exists(file_directory):
#         file_directory = path + str(count)
#         count = count + 1
        
#     # create new directory with filename in /resources/temp/
#     os.makedirs(file_directory)

#     file_location = os.path.join(file_directory, file.filename)

#     # remove the directory
#     background_tasks.add_task(remove_file, file_directory)
#     try:
#         # write uploaded file in file_location
#         with open(file_location, "wb") as file_object:
#             file_object.write(file.file.read())
#         if "classification_pylon" in model_name:
#             print('start')
#             if model_name == "classification_pylon_1024":
#                 result = await pylon_predict(file_location, file.content_type, '1024')
#             elif model_name == "classification_pylon_256":
#                 result = await pylon_predict(file_location, file.content_type, '256')
#             else:
#                 return JSONResponse(content={"success": False, "message": "Model not found"}, status_code=400)
#             # get result (dict) into text file
#             with open(os.path.join(file_directory, 'result', 'prediction.txt'), 'w') as f:
#                 json.dump(result, f)
#             # make zip file of result directory
#             shutil.make_archive(os.path.join(file_directory, file.filename.split('.')[0]),
#                                 'zip',
#                                 root_dir=os.path.join(file_directory, 'result'),
#                                 )
#             print('finish')
#             return FileResponse(os.path.join(file_directory, file.filename.split('.')[0] + '.zip'), status_code=200)
#         elif model_name == "covid19_admission":
#             print('start')
#             result = await covid_predict(file_location, file.content_type)
#             with open(os.path.join(file_directory, 'result', 'prediction.txt'), 'w') as f:
#                 json.dump(result, f)
#             # make zip file of result directory
#             shutil.make_archive(os.path.join(file_directory, file.filename.split('.')[0]),
#                                 'zip',
#                                 root_dir=os.path.join(file_directory, 'result'),
#                                 )
#             print('finish')
#             return FileResponse(os.path.join(file_directory, file.filename.split('.')[0] + '.zip'), status_code=200)
#         else:
#             return JSONResponse(content={"success": False, "message": "Model not found"}, status_code=400)
#     except Exception as e:
#         print(e)
#         return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)

@router.get("/{model_name}/{acc_no}", status_code=200)
async def index(model_name: str, acc_no: str, background_tasks: BackgroundTasks = BackgroundTasks()):

    file_dir = os.path.join(TEMP_DIR, "{}_{}".format(model_name, acc_no))

    # remove the directory
    background_tasks.add_task(remove_file, file_dir)
    try:
        start_time = time.time()
        subprocess.run(["python", os.path.join(BASE_DIR, "pacs_connection", "dicom_function.py"), "-f", "infer", "-a", acc_no, "-m", model_name])
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
        print(e)
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)