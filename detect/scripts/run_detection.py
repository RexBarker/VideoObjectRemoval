# Test image detection implementation
import cv2
from glob import glob
import ObjectDetection.imutils as imu
from ObjectDetection.detect import DetectSingle, TrackSequence, GroupSequence

test_imutils = False
test_single = False 
test_sequence = False 
test_grouping = True
test_maskoutput = True

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

if test_maskoutput:
    groupseq.filter_ObjBBMaskSeq(objNameList='person',)
    groupseq.combine_MaskSequence(writeToDirectory="../data/Colomar/masks/",cleanDirectory=True)


print("done")
