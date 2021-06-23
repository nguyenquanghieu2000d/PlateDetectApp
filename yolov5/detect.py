import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
YOLOV5_PATH = os.getenv("YOLOV5_PATH")
sys.path.insert(0, YOLOV5_PATH)

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class htldYolov5Detect:
    def __init__(self):
        self.source = os.path.join(YOLOV5_PATH, "data/images/1.jpg")
        self.save_path = os.path.join(YOLOV5_PATH, "runs/temp")
        self.weights = os.path.join(YOLOV5_PATH, "MyModel/best.pt")
        self.img_size = 640
        self.conf_thres = 0.30
        self.iou_thres = 0.30
        self.device = ''
        self.view_img = None
        self.save_txt = None
        self.save_conf = None
        self.save_crop = None
        self.nosave = None
        self.classes = None
        self.agnostic_nms = None
        self.augment = None
        self.update = None
        self.project = "runs/detect"
        self.name = "exp"
        self.exist_ok = None
        self.line_thickness = 2
        self.hide_labels = False
        self.hide_conf = False

        # Config

        self.isSetModel = False

    def LoadModel(self, modelPath):
        self.weights = modelPath
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half()
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        self.isSetModel = True

    def detectImage(self, imagePath):
        if not self.isSetModel:
            print("Bạn phải add model vào trước")
            return 1
        if not imagePath:
            print("Bạn phải add image vào trước")
            return 2
        self.source = imagePath

        with torch.no_grad():
            dataset = LoadImages(self.source, img_size=self.img_size, stride=self.stride)
            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = self.model(img, augment=self.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                           agnostic=self.agnostic_nms)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                    s += '%gx%g ' % img.shape[2:]  # print string
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (
                                names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors[c],
                                         line_thickness=self.line_thickness)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')
                    return im0, round(t2 - t1, 2)
            pass

    def detectVideo(self, imagePath, pb=None):
        timeStart = time.time()
        if not self.isSetModel:
            print("Bạn phải add model vào trước")
            return 1
        if not imagePath:
            print("Bạn phải add image vào trước")
            return 2
        self.source = imagePath

        with torch.no_grad():

            # Set Dataloader
            vid_path, vid_writer = None, None

            dataset = LoadImages(self.source, img_size=self.img_size, stride=self.stride)
            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            for path, img, im0s, vid_cap in dataset:
                pb.emit((round((dataset.frame / dataset.nframes) * 100)))
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                t1 = time_synchronized()
                pred = self.model(img, augment=self.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                           agnostic=self.agnostic_nms)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                    s += '%gx%g ' % img.shape[2:]  # print string
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (
                                names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors[c],
                                         line_thickness=self.line_thickness)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')
                    if vid_path != self.save_path:  # new video
                        vid_path = self.save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap is not None:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        vid_writer = cv2.VideoWriter(self.save_path + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                     (w, h))
                    vid_writer.write(im0)

            time.sleep(0.2)
            timeEnd = time.time()
            return self.save_path + ".mp4", round(timeEnd - timeStart, 2)

# a = htldYolov5Detect()
# start = time.time()
# a.LoadModel(modelPath=os.path.join(YOLOV5_PATH, "MyModel/best.pt"))
# end = time.time()
# print(end - start)
# start = time.time()
# img = a.detectVideo(os.path.join(YOLOV5_PATH, "data/images/vid1.mp4"))
# end = time.time()
# print(end - start)

# img = cv2.resize(img, (1000, 1000))
# cv2.imshow("", img)
# cv2.waitKey(0)
