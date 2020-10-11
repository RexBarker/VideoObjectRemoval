import os
import sys
import tempfile
import cv2
from threading import Thread
from time import sleep
from glob import glob

libpath = "/home/appuser/scripts/" # to keep the dev repo in place, w/o linking
sys.path.insert(1,libpath)
import ObjectDetection.imutils as imu
from ObjectDetection.detect import DetectSingle, TrackSequence, GroupSequence
from ObjectDetection.inpaintRemote import InpaintRemote

# ------------
# helper functions

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


# for output bounding box post-processing
#def box_cxcywh_to_xyxy(x):
#    x_c, y_c, w, h = x.unbind(1)
#    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#         (x_c + 0.5 * w), (y_c + 0.5 * h)]
#    return torch.stack(b, dim=1)

#def rescale_bboxes(out_bbox, size):
#    img_w, img_h = size
#    b = box_cxcywh_to_xyxy(out_bbox)
#    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
#    return b


def detect_scores_bboxes_classes(im,model):
    detr.predict(im)
    return detr.scores, detr.bboxes, detr.selClassList 


#def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
#    keep = scores.max(-1).values > confidence
#    scores, boxes = scores[keep], boxes[keep]
#
#    if apply_nms:
#        top_scores, labels = scores.max(-1)
#        keep = batched_nms(boxes, top_scores, labels, iou)
#        scores, boxes = scores[keep], boxes[keep]
#
#    return scores, boxes

def createNullVideo(filePath,message="No Images", heightWidth=(100,100)):
    return imu.createNullVideo(filePath=filePath, message=message, heightWidth=heightWidth)

def testContainerWrite(inpaintObj, workDir=None, hardFail=True):
    # workDir must be accessible from both containers
    if workDir is None:
        workDir = os.getcwd()

    workDir = os.path.abspath(workDir)

    hasErrors = False

    # access from detect container
    if not os.access(workDir,os.W_OK):
        msg = f"Do not have write access to 'detect' container:{workDir}"
        if hardFail: 
            raise Exception(msg)
        else:
            hasErrors = True
            print(msg)

    # access from inpaint container
    inpaintObj.connectInpaint()
    results = inpaintObj.executeCommandsInpaint(
        commands=[f'if [ -w "{workDir}" ]; then echo "OK"; else echo "FAIL"; fi']
    )
    inpaintObj.disconnectInpaint()

    res = [ l.strip() for l in results['stdout'][0]]

    if "FAIL" in res:
        msg = f"Do not have write access to 'inpaint' container:{workdir}" + \
        ",".join([l for l in results['stderr'][0]])
        if hardFail:
            raise Exception(msg)
        else:
            hasErrors = True
            print(msg)

    return not hasErrors 


def performInpainting(detrObj,inpaintObj,workDir,outputVideo, useFFMPEGdirect=False):

    # perform inpainting
    # (write access tested previously)
    workDir = os.path.abspath(workDir)

    with tempfile.TemporaryDirectory(dir=workDir) as tempdir:

        frameDirPath =os.path.join(tempdir,"frames")
        maskDirPath = os.path.join(tempdir,"masks")
        resultDirPath = os.path.join(os.path.join(tempdir,"Inpaint_Res"),"inpaint_res")

        if detrObj.combinedMaskList is None:
            detrObj.combine_MaskSequence()

        detrObj.write_ImageMaskSequence(
            writeImagesToDirectory=frameDirPath,
            writeMasksToDirectory=maskDirPath)

        inpaintObj.connectInpaint()

        trd1 = ThreadWithReturnValue(target=inpaintObj.runInpaint,
                                 kwargs={'frameDirPath':frameDirPath,'maskDirPath':maskDirPath})
        trd1.start() 

        print("working:",end='',flush=True)
        while trd1.is_alive():
            print('.',end='',flush=True)
            sleep(1)

        print("\nfinished")
        inpaintObj.disconnectInpaint()

        stdin,stdout,stderr = trd1.join()
        ok = False
        for l in stdout:
            print(l.strip())
            if "Propagation has been finished" in l: 
                ok = True
        
        assert ok, "Could not determine if results were valid!"
        
        print(f"\n....Writing results to {outputVideo}")

        resultfiles = sorted(glob(os.path.join(resultDirPath,"*.png")))
        imgres = [ cv2.imread(f) for f in resultfiles]
        imu.writeFramesToVideo(imgres, filePath=outputVideo, fps=30, useFFMPEGdirect=True)

        return True


# ***************************
# Model import
# ***************************

# Load detection model
detr = GroupSequence() 
CLASSES = detr.thing_classes
DEVICE = detr.DEVICE 

# load Inpaint remote
inpaint = InpaintRemote()


# The following are imported in app: 
#   >> detect, filter_boxes, detr, transform, CLASSES, DEVICE
