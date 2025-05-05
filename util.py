import numpy as np
import torch
import cv2
#Non-maximum suppression (NMS) is performed on the model predictions
# to remove overlapping redundant boxes and retain only the optimal boxes.
def non_max_suppression(pred, conf_th=0.001, iou_th=0.7):

    import torchvision
    max_det = 300
    max_wh = 7680
    max_nms = 30000
    # Get predictions for a single figure
    pred = pred[0] if isinstance(pred, (list, tuple)) else pred

    bs = pred.shape[0]  # batch size
    nc = pred.shape[1] - 4  # number of classes
    xc = pred[:, 4:(4 + nc)].amax(1) > conf_th

    pred = pred.transpose(-1, -2)
    pred[..., :4] = wh2xy(pred[..., :4])
    # Initialize the output list
    output = [torch.zeros((0, 6), device=pred.device)] * bs
    for xi, x in enumerate(pred):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)

        if nc > 1: # Finding high-confidence locations
            i, j = torch.where(cls > conf_th)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # Best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_th]

        n = x.shape[0]  # number of boxes
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]

        idx = torchvision.ops.nms(boxes, scores, iou_th)  # NMS
        idx = idx[:max_det]

        output[xi] = x[idx]

    return output

def wh2xy(x):
    assert x.shape[-1] == 4, f"expected 4 but input shape is {x.shape}"
    if isinstance(x, torch.Tensor):
        y = torch.empty_like(x, dtype=torch.float32)
    else:
        y = np.empty_like(x, dtype=np.float32)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y
# Drawing blue detection frames and labels with four-corner markings
def draw_box(img, box, label=""):
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)

    if label:
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        h += 3
        outside = y1 < h + 3
        if x1 + w > img.shape[1]:
            x1 = img.shape[1] - w
        y2_label = y1 + h if outside else y1 - h
        cv2.rectangle(img, (x1, y1), (x1 + w, y2_label), (255, 0, 0), -1)

        # Draw corner markers
        corners = [
            ((x1, y1), (x1 + 15, y1), (x1, y1 + 15)),  # Top-left
            ((x2, y2), (x2 - 15, y2), (x2, y2 - 15)),  # Bottom-right
            ((x2, y1), (x2 - 15, y1), (x2, y1 + 15)),  # Top-right
            ((x1, y2), (x1, y2 - 15), (x1 + 15, y2)),  # Bottom-left
        ]
        for center, *lines in corners:
            for pt in lines:
                cv2.line(img, center, pt, (0, 255, 255), 3)

        cv2.putText(img, label, (x1, y1 + h - 3 if outside else y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img
# Draw red warning boxes for highlighting alert objects (e.g., fallers, cars)
def draw_red_box(img, box, label=""):
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, label, (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img
# Calculating the Euclidean distance between two points using NumPy
def calculate_distance_numpy(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))