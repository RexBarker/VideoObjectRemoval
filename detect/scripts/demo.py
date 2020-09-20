# Demo script to process command line level 
# run full video object removal from the command line
import os
import sys
import cv2
import argparse
import tempfile
from glob import glob
import numpy as np
import ObjectDetection.imutils as imu
from ObjectDetection.detect import GroupSequence 
from ObjectDetection.inpaintRemote import InpaintRemote
from threading import Thread

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


# ------------
parser = argparse.ArgumentParser(description='Automatic Video Object Removal')

parser.add_argument('--input',type=str,required=True, 
                    help="Video (.mp4,.avi,.mov) or directory of frames")

parser.add_argument('--start', type=int, required=False, default=None,
                    help="start at frame number")

parser.add_argument('--finish', type=int, required=False, default=None,
                    help="finish at frame number (negative numbers indicate from end)")

parser.add_argument('--outfile',type=str,required=True, 
                    help="Output file (.mp4)")

parser.add_argument('--objlist',type=str,nargs='+', default=None,
                    help='object list, quote delimited per instance: "person:1,5,2" "car:0,2"')

parser.add_argument('--confidence',type=float,default=0.5,
                    help='prediction probablility threshold for object classification (default=0.5)')

parser.add_argument('--minCount', type=int, default=None, 
                    help="minimum length of sequence for object class filtering")

parser.add_argument('--dilationW',type=int, default=0, 
                    help="Use dilation, sets mask half width (default=0, no dilation)")

parser.add_argument('--dilationK',type=str, default='el', choices=['re','cr','el'],
                    help="Use kernel shape elipse(el), cross(cr), or rectangle(re) (default=el)")

parser.add_argument('--useBBmasks', action='store_true', 
                    help="Utilize Bounding Box mask substituion")

parser.add_argument('--annotateOnly', action='store_true',
                    help="Only perform dection and annotates images, skip inpainting")

parser.add_argument('--sequenceOnly', action='store_true',
                    help="Perform detection, sequencing,  skip inpainting")

if __name__ == '__main__':

    #--------------
    # preprocessing

    # get commands
    args = parser.parse_args()

    # check args
    assert sum([args.annotateOnly, args.sequenceOnly]) <= 1, \
        "Ambiguous arguments for 'annotateOnly' and 'sequenceOnly' given"

    # determine number of frames
    vfile = args.input
    assert os.path.exists(vfile), f"Could not determine the input file or directory: {vfile}"

    n_frames = imu.get_nframes(vfile) 
    width,height = imu.get_WidthHeight(vfile) 
    fps = imu.get_fps(vfile)

    # determine number of frames to process
    startframe = 0
    if args.start:
        assert abs(args.start) < n_frames, \
            f"Invalid 'start'={startframe} frame specified, exceeds number of frames ({n_frames})"

        startframe = args.start if args.start >= 0 else n_frames + args.start  # negative indicates from end
    
    finishframe = n_frames
    if args.finish is not None:
        assert abs(args.finish) < n_frames, \
            f"Invalid 'finish'={finishframe} frame specified, exceeds number of frames({n_frames})"

        finishframe = args.finish if args.finish >= 0 else n_frames + args.finish  # negative indicates from end 
    
    assert finishframe > startframe, f"Invalid definition of 'start'={startframe} and 'finish'={finishframe}, start > finish"

    # acquire all frames
    frame_gen = imu.get_frame(vfile,n_frames=n_frames,startframe=startframe, finishframe=finishframe)

    imglist = []
    for img in frame_gen:
        imglist.append(img)
    
    #--------------
    # perform detection, determine number of objects
    objlistDict = {}
    objlistNames = None
    if args.objlist:
        for objnl in args.objlist:
            if ":" in objnl:
                objn,objl = objnl.strip('"\'').replace(" ","").split(":")
                objinds = [ int(v) for v in objl.split(",")]
            else:
                objn = objnl.strip('"\' ')
                objinds = [] 

            if objlistDict.get(objn):
                objlistDict[objn].extend(objinds)
            else:
                objlistDict[objn] = objinds 

        objlistNames = list(objlistDict.keys())

    # intiate engine
    groupseq = GroupSequence(selectObjectNames=objlistNames, score_threshold=args.confidence)
    groupseq.set_imagelist(imglist)

    # perform grouping
    groupseq.groupObjBBMaskSequence(useBBmasks=args.useBBmasks)

    if args.annotateOnly:
        res = groupseq.get_groupedResults()
        annoImages = groupseq.get_annotatedResults()
        imu.writeFramesToVideo(imageList=annoImages,filePath=args.outfile,fps=fps)
        for objn,objl in res.items():
            print(f"ObjName = {objn}, has {len(objl)} instances:")
            for i,obji in enumerate(objl):
                print(f"\t{objn}[{i}] has {len(obji)} frame instances") 
        
        sys.exit(0)

    # filtered by object class, instance, and length 
    if args.minCount is not None: 
        groupseq.filter_ObjBBMaskSeq(allowObjNameInstances=objlistDict,
                                     minCount=args.minCount)

    # fill sequence 
    groupseq.fill_ObjBBMaskSequence(specificObjectNameInstances=objlistDict if objlistDict else None)

    # use dilation
    if args.dilationW > 0:
        groupseq.combine_MaskSequence()
        groupseq.dilateErode_MaskSequence(kernelShape=args.dilationK,
                                          maskHalfWidth=args.dilationW)

    if args.sequenceOnly:
        groupseq.create_animationObject(MPEGfile=args.outfile,
                                        interval=fps, 
                                        useFFMPEGdirect=True)

    #groupseq.fill_ObjBBMaskSequence()






    print("Done")






"""
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

if test_dilateErode:
    detect = DetectSingle(selectObjectNames=['person','car'])
    imgfile = "../data/input.jpg"
    detect.predict(imgfile)
    masks = detect.masks
    mask = imu.combineMasks(masks)

    orig = "original"
    modf = "DilationErosion"
    cv2.namedWindow(orig)
    cv2.namedWindow(modf)
    
    # single dilation operation
    modmask = imu.dilateErodeMask(mask)  
    cv2.imshow(orig,imu.maskToImg(mask))
    cv2.imshow(modf,imu.maskToImg(modmask))
    cv2.waitKey(0)

    modmask = imu.dilateErodeMask(mask,actionList=['dilate','erode','dilate'])  
    cv2.imshow(orig,imu.maskToImg(mask))
    cv2.imshow(modf,imu.maskToImg(modmask))
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if test_sequence:
    fnames = sorted(glob("../data/Colomar/frames/*.png"))[200:400]
    trackseq = TrackSequence(selectObjectNames=['person','car'])
    trackseq.predict_sequence(filelist=fnames)
    res = trackseq.get_sequenceResults()

if test_grouping:
    fnames = sorted(glob("../data/Colomar/frames/*.png"))[200:400]
    groupseq = GroupSequence(selectObjectNames=['person','car'])
    groupseq.load_images(filelist=fnames)
    groupseq.groupObjBBMaskSequence(useBBmasks=test_bbmasks)
    res = groupseq.get_groupedResults(getSpecificObjNames='person')

if test_grouping and test_maskFill:
    groupseq.filter_ObjBBMaskSeq(allowObjNameInstances={'person':[2]},minCount=70)
    #groupseq.fill_ObjBBMaskSequence(specificObjectNameInstances={'person':[0,1,2]})
    groupseq.fill_ObjBBMaskSequence()

if test_grouping and test_maskoutput:
    groupseq.combine_MaskSequence()
    groupseq.dilateErode_MaskSequence(kernelShape='elipse',maskHalfWidth=10)
    groupseq.write_ImageMaskSequence(cleanDirectory=True,
        writeImagesToDirectory="../data/Colomar/fourInpaint/frames",
        writeMasksToDirectory="../data/Colomar/fourInpaint/masks")
    groupseq.create_animationObject(MPEGfile="../data/Colomar/result.mp4")

if test_remoteInpaint:
    rinpaint = InpaintRemote() 
    rinpaint.connectInpaint()

    frameDirPath="/home/appuser/data/Colomar/fourInpaint/frames"
    maskDirPath="/home/appuser/data/Colomar/fourInpaint/masks"

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
"""