import os
import shutil
import sys
from pathlib import Path
import numpy as np
import csv
import cv2
import torch
import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device
from utils.augmentations import letterbox


class Classifier():

    def __init__(self, weights_path, device='', dnn=False, image_size=[416, 416], augment=False, visualize=False,
                 conf_thres=0.005, iou_thres=0.15, max_det=50, classes=None, agnostic_nms=False,  half=False, bs=1):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights_path, device=self.device, dnn=dnn, fp16=half)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.image_size = check_img_size(image_size, s=self.stride)

        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.image_size))  # warmup

        self.augment = augment
        self.visualize = visualize
        self.agnostic_nms = agnostic_nms

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        self.classes = classes

        self.half = half

        self.half &= (self.pt or self.engine) and self.device.type != 'cpu'

        if self.pt:
            self.model.model.half() if self.half else self.model.model.float()

        if self.pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *image_size).to(self.device).type_as(next(self.model.model.parameters())))

    def predict(self, image, image_classification=False):

        # print('detectando lesoes na imagem', image.shape)

        padded_image = letterbox(image, self.image_size, stride=self.stride, auto=True)[0]

        # Convert
        transpose_img = padded_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        continuous_array_image = np.ascontiguousarray(transpose_img)

        continuous_array_image = torch.from_numpy(continuous_array_image).to(self.device)
        continuous_array_image = continuous_array_image.half() if self.half else continuous_array_image.float()  # uint8 to fp16/32
        continuous_array_image /= 255  # 0 - 255 to 0.0 - 1.0

        if len(continuous_array_image.shape) == 3:
            continuous_array_image = continuous_array_image[None]  # expand for batch dim

        pred = self.model(continuous_array_image, augment=self.augment, visualize=self.visualize)

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        results = []

        for i, det in enumerate(pred):

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(continuous_array_image.shape[2:], det[:, :4], image.shape).round()

                # Print results

                # Write results
                for *xyxy, detection_condifence, cls in reversed(det):
                    result = {}

                    class_id = int(cls)  # integer class
                    detection_label = self.names[class_id]
                    # print(class_id, label, p1, p2)

                    result['label'] = detection_label
                    result['confidence'] = detection_condifence.item()

                    top_left_coords = {}
                    top_left_coords['x'] = int(xyxy[0])
                    top_left_coords['y'] = int(xyxy[1])

                    result['topleft'] = top_left_coords

                    bottom_right_coords = {}
                    bottom_right_coords['x'] = int(xyxy[2])
                    bottom_right_coords['y'] = int(xyxy[3])

                    result['bottomright'] = bottom_right_coords

                    results.append(result)

        return results


