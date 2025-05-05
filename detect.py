import cv2
import numpy as np
import torch

import util


class Detector:
    def __init__(self):
        # Load YOLO model weights
        self.model = torch.load('yolo11s.pt', 'cuda')['model'].float()
        self.model.half()
        self.model.eval()
        with open('coco.class', 'r') as f:
            self.classes = f.read().strip().split('\n')

    def detect(self, image):
        # Preprocessing
        shape = image.shape[:2]
        r = 640 / max(shape[0], shape[1])
        if r != 1:
            resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
        height, width = image.shape[:2]

        # Scale ratio (new / old)
        r = min(1.0, 640 / height, 640 / width)

        # Compute padding
        pad = int(round(width * r)), int(round(height * r))
        w = (640 - pad[0]) / 2
        h = (640 - pad[1]) / 2

        if (width, height) != pad:  # resize
            image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

        # Convert HWC to CHW, BGR to RGB
        x = image.transpose((2, 0, 1))[::-1]
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        x = x.unsqueeze(dim=0)
        x = x.cuda()
        x = x.half()
        x = x / 255

        # Model forward inference
        results = self.model(x)
        outputs = util.non_max_suppression(results, 0.15, 0.2)[0]
        # Detection results inverse scaling of the detection frame to restore to original size
        if outputs is not None:
            outputs[:, [0, 2]] -= w
            outputs[:, [1, 3]] -= h
            outputs[:, :4] /= min(height / shape[0], width / shape[1])

            outputs[:, 0].clamp_(0, shape[1])
            outputs[:, 1].clamp_(0, shape[0])
            outputs[:, 2].clamp_(0, shape[1])
            outputs[:, 3].clamp_(0, shape[0])

        return outputs