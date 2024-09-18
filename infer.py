from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('G:/detect/apple-total-total/apple-total-total-original/weights/best.pt')
    model.predict(source='C:/Users/28157/Desktop/image/ooriginal',save=True,save_txt=True)






