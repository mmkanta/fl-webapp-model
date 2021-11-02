from starlette.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTasks
from fastapi import APIRouter, Form, File, UploadFile
from model.classification_pylon_1024.predict_1024 import main as pylon_predict
import shutil

import os
import json

router = APIRouter()

def remove_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)

@router.post("", status_code=200)
def index(model_name: str = Form(...), file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    background_tasks.add_task(remove_file, os.path.join(BASE_DIR, "resources\\temp", file.filename.split('.')[0]))
    try:
        os.makedirs(os.path.join(BASE_DIR, "resources\\temp", file.filename.split('.')[0]), exist_ok=True)

        file_location = os.path.join(BASE_DIR, "resources\\temp", file.filename.split('.')[0], file.filename)
        file_directory = os.path.join(BASE_DIR, "resources\\temp", file.filename.split('.')[0])

        with open(file_location, "wb") as file_object:
            file_object.write(file.file.read())
        if model_name == "classification_pylon_1024":
            result = pylon_predict(file_location, file.content_type)
            with open(os.path.join(file_directory, 'result', 'prediction.json'), 'w') as f:
                json.dump(result, f)
            shutil.make_archive(os.path.join(file_directory, file.filename.split('.')[0]),
                                'zip',
                                root_dir=os.path.join(file_directory, 'result'),
                                )
            
            return FileResponse(os.path.join(file_directory, file.filename.split('.')[0] + '.zip'), status_code=200)
        else:
            return JSONResponse(content={"success": False, "message": "Model not found"}, status_code=400)
    except Exception as e:
        print(e)
        return JSONResponse(content={"success": False, "message": "Internal server error"}, status_code=500)