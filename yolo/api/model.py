from ultralytics import YOLO
import os

model = YOLO('yolov8n-cls.pt')
model.train(data=r'fashion-mnist', epochs = 30, imgsz =28)
