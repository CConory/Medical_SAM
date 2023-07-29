from ultralytics import YOLO
from PIL import Image
import cv2

# model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/weights/yolov8m.pt")
# model = YOLO("/root/autodl-tmp/bowl_2018/yolov8m_300e_dsbowl/weights/best.pt")
# model = YOLO("pretrained/yolov8m_300e_imgsz2560_seq2021/weights/best.pt")
# model = YOLO("pretrained/yolov8m_300e_imgsz2560_seq2021_test/weights/best.pt")
# model = YOLO("pretrained/yolov8m_300e_imgsz20401536_seq2021/weights/best.pt")
model = YOLO("pretrained/yolov8m_300e_imgsz384_CVC/weights/best.pt")

# image_path = "images/dog.jpg"
# image_path = "images/test1.png"
# image_path = "images/9871.bmp"
image_path = "images/120.png"


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# results = model(image, stream=True)
results = model.predict(image, conf=0.001, iou=0.6)
# results = model.predict(image, stream=True, imgsz=2560)
# results = model(image)

# res_plotted = results[0].plot()
# cv2.imshow("result", res_plotted)

boxes_list = []
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    boxes_list.append(boxes.xyxy.tolist())
    
bbox = [[int(i) for i in box] for boxes in boxes_list for box in boxes]
print(bbox)


# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category