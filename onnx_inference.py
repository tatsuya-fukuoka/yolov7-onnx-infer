import os
import time
import logging
import argparse

import cv2

from yolov7.utils import Yolov7onnx


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-mo",
        "--mode",
        type=str,
        default="image",
        help="Inputfile format",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="model/yolov7-tiny.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default='horses.jpg',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-e",
        "--extract",
        type=str,
        default=None,
        help="extract a class",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        action="store_true",
        help="cuda use",
    )
    parser.add_argument(
        "--frame_max",
        type=int,
        default=100,
        help="Maximum number of frames to save in webcam.",
    )
    return parser


def infer_image(args,yolov7):
    img = cv2.imread(args.input_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))
    
    start = time.time()
    result_img = yolov7(img)
    logging.info(f'Infer time: {(time.time()-start)*1000:.2f} [ms]')
    cv2.imwrite(output_path, result_img)
    
    logging.info(f'save_path: {output_path}')
    logging.info(f'Inference Finish!')


def infer_video(args,yolov7):
    cap = cv2.VideoCapture(args.input_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir,os.path.basename(args.input_path))
    
    writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    
    frame_id = 1
    while True:
        ret_val, img = cap.read()
        if not ret_val:
            break
        
        start = time.time()
        result_img = yolov7(img)
        logging.info(f'Frame: {frame_id}/{frame_count}, Infer time: {(time.time()-start)*1000:.2f} [ms]')
        
        writer.write(result_img)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        frame_id+=1
        
    writer.release()
    cv2.destroyAllWindows()
    
    logging.info(f'save_path: {save_path}')
    logging.info(f'Inference Finish!')


def infer_webcam(args,yolov7):
    cap = cv2.VideoCapture(int(args.input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir,'webcam_reslut.mp4')
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(
        save_path, fourcc, fps, (width, height)
        )
    
    frame_id = 1
    while frame_id < args.frame_max:
        ret_val, img = cap.read()
        if not ret_val:
            break
        
        start = time.time()
        result_img = yolov7(img)
        logging.info(f'Infer time: {(time.time()-start)*1000:.2f} [ms]')
        
        #cv2.imshow('windosw', result_img)
        writer.write(result_img)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        frame_id+=1
        
    writer.release()
    cv2.destroyAllWindows()
    
    logging.info(f'save_path: {save_path}')
    logging.info(f'Inference Finish!')


def main():
    args = make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
    
    yolov7 = Yolov7onnx(
        model_path = args.model,
        score_thr = args.score_thr,
        extract_class = args.extract,
        cuda = args.cuda,
    )
    
    if args.mode == 'image':
        infer_image(args,yolov7)
    elif args.mode == 'video':
        infer_video(args,yolov7)
    elif args.mode == 'webcam':
        infer_webcam(args,yolov7)

if __name__ == "__main__": 
    main()
