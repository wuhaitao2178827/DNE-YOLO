from ultralytics import YOLO

model = YOLO('G:/yolov8/ultralytics/cfg/models/v8/yolov8s.yaml')
model.load('G:/yolov8/yolov8s.pt')
model.train(**{'cfg':'G:/yolov8/ultralytics/cfg/default.yaml', 'data':'G:/yolov8/datasets/apple3/data.yaml'})