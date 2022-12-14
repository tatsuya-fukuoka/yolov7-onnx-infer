# yolov7-onnx-infer
Inference with yolov7's onnx model

## 1. Dev Env
```bash
pip install -U pip && pip install -r requirements.txt
```

## 2. Onnx Model Download
```
cd model
sh download_yolov7_tiny_onnx.sh #or download_yolov7_onnx.sh
```

## 3. Inference
### Image
```bash
python onnx_inference.py -i <image_path> -m <onnx_model_path> -e person -s 0.6
```
### Video
```bash
python onnx_inference.py -mo video -i <video_path> -m <onnx_model_path>  -e person -s 0.6
```
## 4. References
* [yolov7](https://github.com/WongKinYiu/yolov7)
* [yolov7/tools/YOLOv7onnx.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb)

## 5. About me
* [Blog](https://chantastu.hatenablog.com/)
* [Twitter](https://twitter.com/chantatsu_blog)
* [Youtube](https://www.youtube.com/channel/UCH9dyrHb8qbEKr_oPsCWq2Q)
