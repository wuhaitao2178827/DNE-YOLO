 
# DNE-YOLO
 
### Abstract
The apple industry, recognized as a pivotal sector in agriculture, increasingly emphasizes the mechanization and intelligent advancement of picking technology. This study innovatively applies a mist simulation algorithm to apple image generation, constructing a dataset of apple images under
mixed sunny, cloudy, drizzling and foggy weather conditions called DNE-APPLE. It introduces a lightweight and efficient target detection network called DNE-YOLO. Building upon the YOLOv8 base model, DNE-YOLO incorporates the CBAM attention mechanism and CARAFE up-sampling
operator to enhance the focus on apples. Additionally, it utilizes GSConv and the dynamic non- monotonic focusing mechanism loss function WIOU to reduce model parameters and decrease reliance on dataset quality. Extensive experimental results underscore the efficacy of the DNE-YOLO model, which achieves a detection accuracy (precision) of 90.7%, a recall of 88.9%, a mean accuracy (mAP50) of 94.3%, a computational complexity (GFLOPs) of 25.4G, and a parameter count of 10.46M across various environmentally diverse datasets. Compared to YOLOv8, it exhibits superior detection accuracy and robustness in sunny, drizzly, cloudy, and misty environments, making it especially suitable for practical applications such as apple picking for agricultural robots. The code for this model is open source at https://github.com/wuhaitao2178827/DNE-YOLO




To install the ultralytics package in developer mode, you will need to have Git and Python 3 installed on your system. Then, follow these steps:

1. Clone the ultralytics repository to your local machine using Git:

    ```bash
    git clone https://github.com/wuhaitao2178827/DNE-YOLO.git
    ```
2. Then you need to go to the root directory and install the required environment.
```bash
cd DNE-YOLO
pip install -r requirements
```
3. The configuration file for our model is in /DNE-YOLO/ultralytics/cfg/models/v8/