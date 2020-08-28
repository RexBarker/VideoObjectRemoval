# Basic image utilities 

import os
from glob import glob

# import some common libraries
import cv2
import numpy as np
from math import log10, ceil

fontconfig = {
    "fontFace"     : cv2.FONT_HERSHEY_SIMPLEX,
    "fontScale"    : 5, 
    "color"        : (0,0,255),
    "lineType"     : 3
}

def bboxToList(bboxTensor):
    return  [float(x) for x in bboxTensor.to('cpu').tensor.numpy().ravel()]

def bboxCenter(bbox):
    """
        Returns (x_c,y_c) of center of bounding box list (x_0,y_0,x_1,y_1)
    """
    return [(bbox[0] + bbox[2])/2,(bbox[1] + bbox[3])/2]

def bboxIoU(boxA, boxB):
    """
        Returns Intersection-Over-Union value for bounding bounding boxA and boxB
        where boxA and boxB are formed by (x_0,y_0,x_1,y_1) lists 
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truthrectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

def bboxToMask(bbox,maskShape):
    """
        Creates a mask(np.bool) based on bbox area for equivalent mask shape
    """
    assert isinstance(bbox,list), "bbox must be list"
    assert isinstance(maskShape,(tuple, list)), "maskShape must be list or tuple"
    x0,y0,x1,y1 = [round(x) for x in bbox]

    bbmask = np.full(maskShape,fill_value=False, dtype=np.bool)
    bbmask[y0:y1,x0:x1] = True
    return bbmask


def combineMasks(maskList):
    """
        Combines the list of masks into a single mask
    """
    # single mask passed
    if not isinstance(maskList,list):
        return maskList   
    elif len(maskList) == 1:
        return maskList[0]     

    maskcomb = maskList[0].copy()
    for msk in maskList[1:]:
        maskcomb = np.logical_or(maskcomb,msk)
    
    return maskcomb


def maskImage(im, mask, mask_color=(0,0,255), inplace=False):
    if inplace:
        outim = im
    else:
        outim = im.copy()
    
    for i in range(3):
        outim[:,:,i] = (mask > 0) * mask_color[i] + (mask == 0) * outim[:, :, i]

    return outim

def maskToImg(mask, toThreeChannel=False):
    """
        converts a mask(dtype=np.bool) to cv2 compatable image (dytpe=np.uint8)
        copies to a 3 channel array if requested
    """
    maskout = np.zeros_like(mask,dtype=np.uint8)
    if mask.dtype == np.bool:
        maskout = np.uint8(255*mask)
    else:
        maskout = np.uint8((mask > 0) * 255) 
    
    if toThreeChannel and len(maskout.shape) == 2:
        mmaskout = np.zeros([*maskout.shape,3],dtype=np.uint8)
        mmaskout[:,:,0] = maskout
        mmaskout[:,:,1] = maskout
        mmaskout[:,:,2] = maskout
        return mmaskout 
    else: 
        return maskout


def dilateErodeMask(mask, actionList=['dilate'], kernelShape='rect', maskHalfWidth=4):
    """
        Dilates or Erodes image mask ('mask') by 'kernelShape', based on mask width
        'maskWidth'= 2 * maskHalfWidth + 1
        'actionList' is a list of actions ('dilate' or 'erode') to perform on the mask
    """
    for act in actionList:
        assert act in ('dilate', 'erode'), "Invalid action specified in actionList"

    if kernelShape.lower().startswith('re'): 
        krnShape = cv2.MORPH_RECT       # rectangular mask
    elif kernelShape.lower().startswith('cr'): 
        krnShape = cv2.MORPH_CROSS      # cross shape
    elif kernelShape.lower().startswith('el'):
        krnShape = cv2.MORPH_ELLIPSE    # elliptical shape (or circlular)
    else:
        raise Exception(f"Unknown kernel mask shape specified: {kernelShape}")

    assert maskHalfWidth > 0, "Error: maskHalfWidth must be > 0" 

    maskWasDtype = mask.dtype
    maskWidth = 2 * maskHalfWidth + 1
    krnElement = cv2.getStructuringElement(krnShape, 
                                           (maskWidth,maskWidth),  
                                           (maskHalfWidth, maskHalfWidth))

    maskout = np.uint8(mask.copy())
    for act in actionList:
        if act == 'dilate': 
            maskout = cv2.dilate(maskout,krnElement)
        elif act == 'erode': 
            maskout = cv2.erode(maskout,krnElement)
        else:
            pass  # hmm, shouldn't get here

    maskout.dtype = maskWasDtype
    return maskout


def writeImagesToDirectory(imageList,dirPath,minPadLength=None,imgtype='png',cleanDirectory=False):
    """
        writes flat list of image arrays to directory
        Here, it is understood that images are an np.array, dtype='uint8' 
        of shape (w,h,3)
    """
    assert imgtype in ('png', 'jpg'), f"Invalid image type '{imgtype}' given"

    if not os.path.isdir(dirPath):
        path = ''
        for d in dirPath.split('/'):
            if not d: continue
            path += d + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
    elif cleanDirectory:
        for f in glob(os.path.join(dirPath,"*." + imgtype)):
            os.remove(f) # danger Will Robinson

    n_frames = len(imageList)
    padlength = ceil(log10(n_frames)) if minPadLength is None else minPadLength    
    for i,img in enumerate(imageList):
        fname = str(i).rjust(padlength,'0') + '.' + imgtype
        fname = os.path.join(dirPath,fname)
        cv2.imwrite(fname,img)
    
    return n_frames


def writeMasksToDirectory(maskList,dirPath,minPadLength=None,imgtype='png',cleanDirectory=False):
    """
        writes flat list of mask arrays to directory
        Here, it is understood that mask is an np.array,dtype='bool'
        of shape (w,h), will be output to (w,h,3) for compatibility
    """
    assert imgtype in ('png', 'jpg'), f"Invalid image type '{imgtype}' given"

    if not os.path.isdir(dirPath):
        path = ''
        for d in dirPath.split('/'):
            if not d: continue
            path += d + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
    elif cleanDirectory:
        for f in glob(os.path.join(dirPath,"*." + imgtype)):
            os.remove(f) # danger Will Robinson

    n_frames = len(maskList)
    padlength = ceil(log10(n_frames)) if minPadLength is None else minPadLength    
    for i,msk in enumerate(maskList):
        fname = str(i).rjust(padlength,'0') + '.' + imgtype
        fname = os.path.join(dirPath,fname)
        cv2.imwrite(fname,msk * 255)
    
    return n_frames


def writeFramesToVideo(imageList,filePath,fps=30):
    """
        Writes given set of frames to video file (platform specific coding)
        format is 'mp4' or 'avi'
    """
    assert len(imageList) > 1, "Cannot make video with single frame"
    height,width =imageList[0].shape[:2]

    dirPath = os.path.dirname(filePath)
    if not os.path.isdir(dirPath):
        path = ''
        for d in dirPath.split('/'):
            if not d: continue
            path += d + '/'
            if not os.path.isdir(path):
                os.mkdir(path)

    if filePath.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif filePath.endswith(".avi"):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        assert False, f"Could not determine the video output type from {filePath}"
        
    outvid = cv2.VideoWriter(filePath, fourcc, fps, (width,height) )

    # write out frames to video
    for im in imageList:
        outvid.write(im)

    outvid.release()

    return len(imageList)


def createNullVideo(filePath,message="No Image",heightWidth=(100,100)):
    h,w = heightWidth
    imgblank = np.zeros((h,w,3),dtype=np.uint8)
    if message:
        imgblank = cv2.putText(imgblank,message,(h // 2, w // 2),**fontconfig)

    # create blank video with 2 frames
    return writeFramesToVideo([imgblank,imgblank],filePath=filePath,fps=1)
    

def maskedItemRelativeHistogram(img, msk,n_bins=10):
    im = img.copy()
    
    # reduce image to masked portion only 
    for i in range(3):
        im[:,:,i] = (msk == 0) * 0 + (msk > 0) * im[:, :, i]
    
    take_ys= im.sum(axis=2).mean(axis=1) > 0
    take_xs= im.sum(axis=2).mean(axis=0) > 0
    
    imsub=im[take_ys,:,:]
    imsub= imsub[:,take_xs,:]
    
    # determine average vectors for each direction
    h_av = np.mean((imsub == 0) * 0 + (imsub > 0) * imsub,axis=1)
    v_av = np.mean((imsub == 0) * 0 + (imsub > 0) * imsub,axis=0)
    
    #h_abs_vec = np.array(range(h_av.shape[0]))/h_av.shape[0]
    h_ord_vec = h_av.sum(axis=1)/h_av.sum(axis=1).max()
    
    #v_abs_vec = np.array(range(v_av.shape[0]))/v_av.shape[0]
    v_ord_vec = v_av.sum(axis=1)/v_av.sum(axis=1).max()
    
    h_hist=np.histogram(h_ord_vec,bins=n_bins)
    v_hist=np.histogram(v_ord_vec,bins=n_bins)
    
    return (h_hist[0]/h_hist[0].sum(), v_hist[0]/v_hist[0].sum())
    

def drawPoint(im, XY, color=(0,0,255), radius=0, thickness = -1, inplace=False):
    """
        draws a points over the top of an image 
        point : (x,y) 
        color : (R,G,B)
    """
    xy = tuple([round(v) for v in XY])
    if inplace:
        outim = im
    else:
        outim = im.copy()

    outim = cv2.circle(outim, xy, radius=radius, color=color, thickness=thickness)
    return outim


def drawPointList(im, XY_color_list, radius=0, thickness = -1, inplace=False):
    """
        draws points over the top of an image given a list of (point,color) pairs
        point : (x,y) 
        colors : (R,G,B)
    """
    if inplace:
        outim = im
    else:
        outim = im.copy()

    for XY,color in XY_color_list[:-1]:
        xy = tuple([round(v) for v in XY])
        outim = cv2.circle(outim, xy, radius=radius, color=color, thickness=round(thickness * 0.8))
    
    # last point is larger
    XY,color = XY_color_list[-1]
    xy = tuple([round(v) for v in XY])
    outim = cv2.circle(outim, xy, radius=radius, color=color, thickness=round(thickness))
        
    return outim
    
if __name__ == "__main__":
    pass