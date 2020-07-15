# Test image detection implementation
import cv2
from glob import glob
import ObjectDetection.imutils as imu
from ObjectDetection.detect import DetectSingle, TrackSequence

test_imutils = False
test_single = True 
test_sequence = True

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
    cv2.destroyAllWindows()

if test_sequence:
    fnames = sorted(glob("../data/Colomar/frames/*.png"))[100:200]
    trackseq = TrackSequence(selectObjectNames=['person','car'])
    trackseq.load_images(filelist=fnames)
    res = trackseq.get_results()

print("done")
