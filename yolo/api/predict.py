from ultralytics import YOLO
import cv2

model = YOLO('./runs/classify/train10/weights/last.pt')
image_path = r'./test_image/OIP.jpg'
input_image = cv2.imread(image_path)
# input_image = cv2.resize(input_image, (28,28))
results = model.predict(input_image)

for result in results:
    #shows the confidence scores for class IDs

    result.show()
