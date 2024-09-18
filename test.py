from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('/root/yolov8/runs/detect/apple-total-total-CBAM+slimneck+WIOU-1/weights/best.pt')
    model.val(**{'data': '/root/autodl-tmp/apple/data.yaml', 'split':'test'})





