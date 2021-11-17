from fastapi import FastAPI
# from pydantic import BaseModel
import uvicorn

from api import infer

app = FastAPI()
@app.get("/")
def index():
    return {"Hello": "World"}

app.include_router(
    infer.router,
    prefix="/api/infer",
    tags=["Inference"],
    responses={404: {"success": False, "message": "Not found"}},
)

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=7000)

# uvicorn main:app --reload --port 7000
# uvicorn main:app --port 7000 --workers 2
# curl -X POST http://localhost:7000/api/infer -H  "Content-Type: multipart/form-data" -F "model_name=covid19_admission" -F file="@C:\Users\USER\Desktop\Onnx_Inference_example\0041018.dcm" --output "test1234.zip"
# docker run --rm -it -p 80:8000 test_py -> localhost