# Test image detection implementation
import cv2
from time import time, sleep
from glob import glob
import ObjectDetection.imutils as imu
from ObjectDetection.detect import DetectSingle, TrackSequence, GroupSequence
from ObjectDetection.inpaintRemote import InpaintRemote
from threading import Thread

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

test_imutils = False
test_single = False 
test_sequence = False 
test_grouping = False
test_maskoutput = False
test_remoteInpaint = True

if test_imutils:
    bbtest = [0.111, 0.123, 0.211, 0.312]
    bbc = imu.bboxCenter(bbtest)
    print(bbc)

if test_single:
    detect = DetectSingle(selectObjectNames=['person','car'])
    imgfile = "../data/input.jpg"
    detect.predict(imgfile)
    imout = detect.annotate()

    cv2.imshow('results',imout)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    imout = detect.visualize_all(scale=1.2)
    cv2.imshow('results',imout)
    cv2.waitKey(0)
    cv2.destroyAllWindows(); 

if test_sequence:
    fnames = sorted(glob("../data/Colomar/frames/*.png"))[200:400]
    trackseq = TrackSequence(selectObjectNames=['person','car'])
    trackseq.predict_sequence(filelist=fnames)
    res = trackseq.get_sequenceResults()

if test_grouping:
    fnames = sorted(glob("../data/Colomar/frames/*.png"))[200:400]
    groupseq = GroupSequence(selectObjectNames=['person','car'])
    groupseq.load_images(filelist=fnames)
    groupseq.groupObjBBMaskSequence()
    res = groupseq.get_groupedResults(getSpecificObjNames='person')

if test_grouping and test_maskoutput:
    groupseq.filter_ObjBBMaskSeq(objNameList='person',)
    groupseq.combine_MaskSequence(cleanDirectory=True,
        writeImagesToDirectory="../data/Colomar/toInpaint/frames",
        writeMasksToDirectory="../data/Colomar/toInpaint/masks")

if test_remoteInpaint:
    rinpaint = InpaintRemote() 
    rinpaint.connectInpaint()

    frameDirPath="/home/appuser/data/Colomar/toInpaint/frames"
    maskDirPath="/home/appuser/data/Colomar/toInpaint/masks"

    trd1 = ThreadWithReturnValue(target=rinpaint.runInpaint,
                                 kwargs={'frameDirPath':frameDirPath,'maskDirPath':maskDirPath})
    trd1.start() 

    print("working:",end='',flush=True)
    while trd1.is_alive():
        print('.',end='',flush=True)
        sleep(1)

    print("\nfinished")
    rinpaint.disconnectInpaint()

    stdin,stdout,stderr = trd1.join()
    ok = False
    for l in stdout:
        if "Propagation has been finished" in l: 
            ok = True
        print(l.strip())

print("done")
