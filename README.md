# yolov7-onnx-infer
Inference with yolov7's onnx model

## Dev Env
```bash
pip install -U pip && pip install -r requirements.txt
```

## Onnx Model Download
```
cd model
sh download_yolov7_tiny_onnx.sh #or download_yolov7_onnx.sh
```

## Inference
### Image
```bash
python onnx_inference.py -i <image_path> -m <onnx_model_path>
```
### Video
```bash
python onnx_inference.py -mo video -i <video_path> -m <onnx_model_path>
```
