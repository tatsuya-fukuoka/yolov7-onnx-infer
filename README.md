# yolov7-onnx-infer
Inference with yolov7's onnx model

## onnx model
* [yolov7-tiny.onnx](https://drive.google.com/file/d/1-P3RpFnbXv_a99EW_wiBCUcZ7ssJzEnj/view?usp=share_link)
* [yolov7.onnx](https://drive.google.com/file/d/1pL-VhELoJMIwT5H9hGi5y0wgztfb0CsI/view?usp=share_link)

## 1. Dev Env
### 1.1 pip install
```bash
pip install -U pip && pip install -r requirements.txt
```
### 1.2 Docker
#### Dockerfile
Ubuntu 20.04
```Dockerfile
FROM ubuntu:20.04
USER root

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

LABEL version="1.0"
LABEL description="Build operating environment for yolov7-onnx-infer"

RUN apt-get update && \
    apt-get -y install python3-pip && \
    apt-get -y install git && \
    apt-get -y install libgl1-mesa-dev && \
    apt-get -y install libglib2.0-0 && \
    pip install -U pip && \
    pip install opencv-python onnxruntime
```
Debian
```Dockerfile
FROM debian:stable-slim
USER root

LABEL version="1.0"
LABEL description="Build operating environment for yolov7-onnx-infer"

RUN apt-get update && \
    apt-get -y install python3-pip && \
    apt-get -y install git && \
    apt-get -y install libgl1-mesa-dev && \
    apt-get -y install libglib2.0-0 && \
    pip install -U pip && \
    pip install opencv-python onnxruntime
```
#### docker build
```bash
docker build -t tatsuya060504/yolov7-onnx-infer:v1.0.0 .
```
#### docker pull
[Docker Hub](https://hub.docker.com/repository/docker/tatsuya060504/yolov7-onnx-infer)
```bash
docker pull tatsuya060504/yolov7-onnx-infer:raspberrypi
```
docker run
```bash
docker run -it --name=yolov7-onnx-infer -v $(pwd):/home tatsuya060504/yolov7-onnx-infer:raspberrypi
```

## 2. Onnx Model Download
You can download the yolox-onnx model by executing the shell script in the model folder.
```
cd model
sh download_yolov7_tiny_onnx.sh #or download_yolov7_onnx.sh
```

## 3. Inference
### Image
```bash
python onnx_inference.py -i <image_path> -m <onnx_model_path> -e <class_name> -s <score_threshold>
```
### Video
```bash
python onnx_inference.py -mo video -i <video_path> -m <onnx_model_path> -e <class_name> -s <score_threshold>
```
## 4. References
* [yolov7](https://github.com/WongKinYiu/yolov7)
* [yolov7/tools/YOLOv7onnx.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb)

## 5. About me
* [Blog](https://chantastu.hatenablog.com/)
* [Twitter](https://twitter.com/chantatsu_blog)
* [Youtube](https://www.youtube.com/channel/UCH9dyrHb8qbEKr_oPsCWq2Q)
