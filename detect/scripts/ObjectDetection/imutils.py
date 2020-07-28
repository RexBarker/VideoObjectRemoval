# Basic image utilities 

import os
from glob import glob

# import some common libraries
import cv2
import numpy as np
from math import log10, ceil

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


def combineMasks(maskList):
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

def writeMasksToDirectory(maskList,dirPath,imgtype='png',cleanDirectory=False):
    """
        writes flat list of mask arrays to directory
        Here, it is understood that mask is an np.array,dtype='bool'
        of shape (w,h)
    """
    assert imgtype in ('png', 'jpg'), f"Invalid image type '{imgtype}' given"

    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)
    elif cleanDirectory:
        for f in glob(os.path.join(dirPath,"*." + imgtype)):
            pass
            #os.remove(f) # danger Will Robinson

    n_frames = len(maskList)
    padlength = ceil(log10(n_frames))    
    for i,msk in enumerate(maskList):
        fname = str(i).rjust(padlength,'0') + '.' + imgtype
        fname = os.path.join(dirPath,fname)
        cv2.imwrite(fname,msk * 255)
    
    return n_frames


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