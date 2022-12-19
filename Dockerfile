FROM debian:stable-slim

LABEL version="1.0"
LABEL description="yolov7をonnxモデルで実行"

RUN apt-get update
RUN apt-get -y install pip
RUN apt-get -y install libgl1-mesa-dev && apt-get -y install libglib2.0-0
RUN pip install -U pip
RUN pip install --no-cache-dir onnxruntime==1.13.1 opencv-python==4.6.0.66 filterpy
