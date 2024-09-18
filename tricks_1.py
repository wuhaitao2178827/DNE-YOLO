import os, cv2, tqdm, shutil
import numpy as np

def xywh2xyxy(box):
    box[:, 0] = box[:, 0] - box[:, 2] / 2
    box[:, 1] = box[:, 1] - box[:, 3] / 2
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]
    return box

def iou(box1, box2):
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)

    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))
 
    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))
 
    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter
 
    iou = area_inter / area_union
    return iou


def draw_box(img, box, label, color):
    # 根据标签确定文字
    text = classes[int(label[0])] if label is not None else "miss"

    # 绘制框
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=10)

    # 在框的左上角输出类别名称或"miss"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 10
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # 调整文本框的大小
    text_width, text_height = text_size[0], text_size[1]
    text_bg_width, text_bg_height = text_width + 20, text_height + 20

    text_x = int(box[0])
    text_y = int(box[1]) - 5 - text_bg_height

    # 绘制文本背景
    cv2.rectangle(img, (text_x, text_y), (text_x + text_bg_width, text_y + text_bg_height), color, -1)

    # 在文本背景上绘制文本
    cv2.putText(img, text, (text_x + 5, text_y + text_bg_height - 5), font, font_scale, (255, 255, 255), font_thickness)

    return img


if __name__ == '__main__':
    postfix = 'jpg'
    img_path = 'C:/Users/28157/Desktop/look/image'
    label_path = 'C:/Users/28157/Desktop/look/label/1'
    predict_path = 'C:/Users/28157/Desktop/look/label/yolov8'
    save_path = 'C:/Users/28157/Desktop/look/yolov8'
    classes = ['apple']
    detect_color, missing_color = (0, 0, 255), (0, 255, 0)
    iou_threshold = 0.45

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    all_right_num, all_missing_num, all_error_num = 0, 0, 0
    with open('result.txt', 'w') as f_w:
        for path in tqdm.tqdm(os.listdir(label_path)):
            image = cv2.imread(f'{img_path}/{path[:-4]}.{postfix}')
            if image is None:
                print(f'image:{img_path}/{path[:-4]}.{postfix} not found.', file=f_w)
            h, w = image.shape[:2]

            try:
                with open(f'{predict_path}/{path}') as f:
                    pred = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float32), f.readlines())))
                    pred[:, 1:5] = xywh2xyxy(pred[:, 1:5])
                    pred[:, [1, 3]] *= w
                    pred[:, [2, 4]] *= h
                    pred = list(pred)
            except:
                pred = []

            try:
                with open(f'{label_path}/{path}') as f:
                    label = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float32), f.readlines())))
                    label[:, 1:] = xywh2xyxy(label[:, 1:])
                    label[:, [1, 3]] *= w
                    label[:, [2, 4]] *= h
            except:
                print(f'label path:{label_path}/{path} (not found or no target).', file=f_w)

            right_num, missing_num, error_num = 0, 0, 0
            label_id, pred_id = list(range(label.shape[0])), [] if len(pred) == 0 else list(range(len(pred)))
            for i in range(label.shape[0]):
                if len(pred) == 0: break
                ious = iou(label[i:i + 1, 1:], np.array(pred)[:, 1:5])[0]
                ious_argsort = ious.argsort()[::-1]
                missing = True
                for j in ious_argsort:
                    if ious[j] < iou_threshold: break
                    if label[i, 0] == pred[j][0]:
                        image = draw_box(image, pred[j][1:5],label[i], detect_color)
                        pred.pop(j)
                        missing = False
                        right_num += 1
                        break

                if missing:
                    image = draw_box(image, label[i][1:5],None, missing_color)
                    missing_num += 1



            all_right_num, all_missing_num, all_error_num = all_right_num + right_num, all_missing_num + missing_num, all_error_num + error_num
            cv2.imwrite(f'{save_path}/{path[:-4]}.{postfix}', image)
            print(f'name:{path[:-4]} right:{right_num} missing:{missing_num} error:{error_num}', file=f_w)
        print(f'all_result: right:{all_right_num} missing:{all_missing_num} error:{all_error_num}', file=f_w)