import time

import cv2
import onnxruntime as ort
import random
import numpy as np

from yolov7.coco_classes import COCO_CLASSES
from yolov7.color_list import _COLORS


class YOLOV7ONNX(object):
    def __init__(
        self,
        model_path,
        cuda
    ):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
    
    def inference(self, img, timer, args):
        timer.tic()
        outname, inp, ratio, dwdh = self.preproc(img)
        outputs = self.session.run(outname, inp)[0]
        timer.toc()
        reslut_img = self.visual(img, outputs, ratio, dwdh, args)
        return reslut_img

    def preproc(self,img):
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape
        
        outname = [i.name for i in self.session.get_outputs()]

        inname = [i.name for i in self.session.get_inputs()]

        inp = {inname[0]:im}
        
        return outname, inp, ratio, dwdh
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)
    
    def visual(self, img, outputs, ratio, dwdh, args):
        ori_images = [img.copy()]
        
        if args.extract is not None:
            extract_cls_i = COCO_CLASSES.index(args.extract)
            
            for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
                if int(cls_id) == extract_cls_i:
                    if args.score_thr is not None:
                        if score > args.score_thr:
                            self.vis(ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh, args)
                    else:
                        self.vis(ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh, args)
        else:
            for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
                if args.score_thr is not None:
                    if score > args.score_thr:
                        self.vis(ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh, args)
                else:
                    self.vis(ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh, args)
            
        return ori_images[0]
    
    def vis(self, ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh, args):
        image = ori_images[int(batch_id)]
        
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        
        #class_name list
        names = list(COCO_CLASSES)
        
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.rectangle(
            image,
            (box[0], box[1] + 1),
            (box[0] + txt_size[0] + 1, box[1] + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(image,text,(box[0], box[1] + txt_size[1]),font,0.4,txt_color,thickness=1)
