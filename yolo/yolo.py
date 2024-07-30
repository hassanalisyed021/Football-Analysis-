from ultralytics import YOLO
import cv2

model = YOLO('models\\best.pt')
result = model("input_videos\\sample 2.mp4",save = True)
print(result[0])
print("###########################################")