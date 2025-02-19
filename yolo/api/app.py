from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
model_path = r'yolo/models/yolov8n-cls.pt'
model = YOLO(model_path)

class Image(BaseModel):
    img_url : str

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post('/classify')
async def classify(image: Image):

    image_url = image.img_url

    output = model(image_url)[0]

    pred = output.probs.data.tolist()

    pred_index = int(np.argmax(pred))

    return {'prediction' : output.names[pred_index],
            'Score': pred[pred_index],
            'image_url' : image_url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
