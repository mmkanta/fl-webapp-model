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
    uvicorn.run('main:app', host="localhost", port=7000)

# uvicorn main:app --reload --port 7000
# uvicorn main:app --port 7000 --workers 2