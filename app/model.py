import os
import sys

libpath = "/home/appuser/scripts/" # to keep the dev repo in place, w/o linking
sys.path.insert(1,libpath)
import ObjectDetection.imutils as imu
from ObjectDetection.detect import DetectSingle, TrackSequence, GroupSequence


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect_scores_bboxes_classes(im,model):
    detr.predict(im)
    return detr.scores, detr.bboxes, detr.selClassList 


def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]

    return scores, boxes

def createNullVideo(filePath,message="No Images", heightWidth=(100,100)):
    return imu.createNullVideo(filePath=filePath, message=message, heightWidth=heightWidth)


# COCO classes
#CLASSES = [
#    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
#    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
#    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#    'toothbrush'
#]


# Load model
detr = GroupSequence() 
#detr = DetectSingle() 
CLASSES = detr.thing_classes
DEVICE = detr.DEVICE 
#detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
#detr.eval().to(DEVICE)

# standard PyTorch mean-std input image normalization
#transform = T.Compose([
#    T.Resize(500),
#    T.ToTensor(),
#    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#])


# The following are imported in app: 
#   >> detect, filter_boxes, detr, transform, CLASSES, DEVICE
