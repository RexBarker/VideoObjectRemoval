# Detection to object instances

# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import os
from glob import glob
import detectron2

# import some common libraries
import numpy as np
import pandas as pd
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# module specific library
import ObjectDetection.imutils as imu

class DetectSingle:
    def __init__(self, 
                 score_threshold = 0.5,
                 model_zoo_config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                 selectObjectNames = None,  #default to all
                 **kwargs):
        # engine specific variables
        self.outputs = None
        self.cfg = None
        self.predictor = None
        self.things = None

        # instance specific variables
        self.im = None
        self.selClassList = []
        self.selObjNames = selectObjectNames
        self.selObjIndices = []


        # initialize engine
        self.__initEngine(score_threshold, model_zoo_config_path)

        # annotation configuration
        self.fontconfig = { 
            "fontFace"     : cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale"    : 1, 
            "color"        : (0,255,0),
            "lineType"     : 3
        }
    
    def __initEngine(self, score_threshold, model_zoo_config_path):
        #initialize configuration
        self.cfg = get_cfg()

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file((model_zoo_config_path)))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold  # score threshold to consider object of a class

        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_config_path)

        self.predictor = DefaultPredictor(self.cfg)
        self.things = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.thing_classes = self.things.thing_classes
        self.thing_colors = self.things.thing_colors

        if self.selObjNames is None:
            self.selObjNames = self.thing_classes
            self.selObjIndices = list(range(len(self.selObjNames)))
        else:
            for n in self.selObjNames:
                assert n in self.thing_classes, f"Error finding object class name: {n}"
                self.selObjIndices.append(self.selObjNames.index(n))                

    def predict(self,img,selObjectNames=None):
        if isinstance(img,str):
            # was an image file path 
            assert os.path.exists(img), f"Specified image file {img} does not exist"
            self.im = cv2.imread(img)
        elif isinstance(img,np.ndarray):
            self.im = img
        else:
            raise Exception("Could not determine the object instance of 'img'")

        if selObjectNames is None:
            selObjectNames = self.selObjNames

        outputs = self.predictor(self.im)
        classes = list(outputs['instances'].pred_classes.cpu().numpy()) 
        objects = [self.thing_classes[c] in selObjectNames for c in classes]

        self.outputs = outputs
        self.masks = [ outputs['instances'].pred_masks[i].cpu().numpy() for i,o in enumerate(objects) if o ]
        self.bboxes = [ imu.bboxToList(outputs['instances'].pred_boxes[i]) for i,o in enumerate(objects) if o ]
        self.selClassList = [ classes[i] for i,o in enumerate(objects) if o ]
    
    def get_results(self,getImage=True, getMasks=True, getBBoxes=True, getClasses=True):
        res = dict()
        if getImage: res['im'] = self.im
        if getMasks: res['masks'] = self.masks
        if getBBoxes: res['bboxes'] = self.bboxes
        if getClasses: res['classes'] = self.selClassList

        return res 

    def annotate(self, im=None, addIndices=True):
        """
            Adds annotation of the selected instances to the image 
            Indices are added according to the order of prediction 
        """
        if im is None: 
            im = self.im

        outim = im.copy()

        for i,(msk,bbox) in enumerate(zip(self.masks,self.bboxes)):
            color = self.thing_colors[i]
            x,y = [round(c) for c in imu.bboxCenter(bbox)]
            outim = imu.maskImage(outim,msk,color)
            if addIndices:
                cv2.putText(outim,str(i), (x,y), **self.fontconfig)
        
        return outim
    
    def visualize_all(self,im=None, scale=1.0):
        """
            Adds full annotation to all objects within the image, based
            upon the detectron2 specification
        """
        if im is None: 
            im = self.im

        outim = im.copy()
        # assume that im is opencv format (BGR), so reverse
        vout =  Visualizer(outim[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=scale)
        vout = vout.draw_instance_predictions(self.outputs["instances"].to("cpu"))

        return vout.get_image()[:, :, ::-1]  # reverses channels again

# Class: TrackSequence
# - creates the basic sequence collection for set of input frames
# - defined by either the filelist or a fileglob (list is better, you do the picking)
# - assumptions are that all frames will be sequenced as XXX.jpg or XXX.png
# - where XXX is the numerical order of the frame in the sequence
class TrackSequence(DetectSingle):
    def __init__(self, *args, **kwargs):
        super(TrackSequence,self).__init__(*args, **kwargs)

        # sequence tracking variables 
        self.selectFiles = None
        self.imglist = []
        self.masklist = []
        self.bboxlist = []
        self.objclasslist = []

    def load_images(self,fileglob=None, filelist=None):
        if fileglob is not None:
            files = sorted(glob(fileglob))
        elif filelist is not None:
            files = filelist 
            # check if these files really existed 
            # (this doesn't make sense for globs, since they were checked to make the glob)
            checkfiles = [ os.path.exists(f) for f in files]
            if not all(checkfiles):
                mfiles = ",".join([f for f,e in zip(files,checkfiles) if not e])
                raise Exception("Missing files in list: " + mfiles)
        else:
            raise Exception("No filelist or fileglob was supplied")

        # load images using OpenCV
        for fname in files:
            im = cv2.imread(fname)
            self.imglist.append(im)
        
        return len(self.imglist)
    
    def get_images(self):
        return self.imglist 

    def predict_sequence(self,fileglob=None, filelist=None, **kwargs):
        if len(self.imglist) == 0:
            # load images was not called yet
            self.load_images(fileglob=fileglob, filelist=filelist, **kwargs)

        for im in self.imglist:
            self.predict(im)  
            self.masklist.append(self.masks)
            self.bboxlist.append(self.bboxes)
            self.objclasslist.append(self.selClassList)
        
        return len(self.masklist) 

    def get_sequenceResults(self,getImage=True, getMasks=True, getBBoxes=True, getClasses=True):
        """
            Results for sequence prediction, returned as dictionary for objName
            (easily pickelable)
        """
        res = dict()
        if getImage: res['im'] = self.imglist
        if getMasks: res['masks'] = self.masklist
        if getBBoxes: res['bboxes'] = self.bboxlist
        if getClasses: res['classes'] = self.objclasslist

        return res  

# Class GroupSequence
# - creates the grouping of a sequence by object based on class
# - Object (in this sense): A single person, car, truck, teddy bear, etc.
# - Class (in this sense): The type of object (i.e. persons, cars, trucks, teddy bears, etc.)
# Main output:
# - "objBBMaskSeq": { 
#                     objName1: [ [[seqBB], [seqMask] (obj1)], [[seqBB], [seqMask] (obj2)], etc]
#                     objName2: [ [[seqBB], [seqMask] (obj1)], [[seqBB], [seqMask] (obj2)], etc]
#                    }
class GroupSequence(TrackSequence):
    def __init__(self, *args, **kwargs):
        super(GroupSequence,self).__init__(*args, **kwargs)

        # sequence tracking variables 
        self.objBBMaskSeqDict = None
        self.objBBMaskSeqGrpDict = None
        self.combinedMaskList = None

    @staticmethod
    def __assignBBMaskToGroupByDistIndex(attainedGroups, trialBBs, trialMasks, index=None, widthFactor=2.0):
        """
            Assign points to grouping:
            - [trialBBs] list of bounding boxs in given frame 
            - [trialMasks] list of Masks in given frame 
            - index : the index of the frame in the sequence
            - Note: len(trialBBs) == len(trialMasks), if nothing is predicted, an empty list is a place holder
            This function is meant to be called recursively, adding to its previous description of attainedGroups
        """
        
        # trivial case.  No points to work on
        if trialBBs is None or not len(trialBBs):
            return attainedGroups
        
        # no actual groups assigned yet, so enumerate the existing points into groups
        if attainedGroups is None or not len(attainedGroups):
            return [ [[bb,msk,index]] for bb,msk in zip(trialBBs,trialMasks)]
        
        lastGrpBBMsk = [grp[-1][:2] for grp in attainedGroups]
        currgrps = { i for i in range(len(attainedGroups))}
        
        for bbx,msk in zip(trialBBs,trialMasks):
            x1,y1,x2,y2 = bbx
            w = x2 - x1
            h = y2 - y1
            xc = (x1+x2)/2
            yc = (y1+y2)/2

            dist = []
            for gi,gbbmsk in enumerate(lastGrpBBMsk):
                gbb,gmsk = gbbmsk
                gx1,gy1,gx2,gy2 = gbb
                gxc = (gx1+gx2)/2
                gyc = (gy1+gy2)/2
                d = np.sqrt((xc - gxc)**2 + (yc - gyc)**2)
                dist.append([d,gi])
            
            #remove possible groups which were previously found
            dist = [[di,gi] for di,gi in dist if gi in currgrps]
        
            dist0 = dist[0] if dist else None
            mdist = dist0[0] if dist0 else None
            mdist_idx = dist0[1] if dist0 else None

            if len(dist) > 1:
                for d,idx in dist[1:]:
                    if d < mdist:
                        mdist = d
                        mdist_idx = idx
            
            if mdist is None or mdist > widthFactor * w:
                # must be a new group
                attainedGroups.append([[bbx,msk,index]])  
            else:
                # belongs to an existing group
                attainedGroups[mdist_idx].append([bbx,msk,index])
                #currgrps.remove(mdist_idx) #--> cleanout
                
        return attainedGroups

    
    def __createObjBBMask(self):
        assert self.objclasslist is not None, "No objclass sequences exist"
        assert self.imglist is not None, "No image sequences exist"
        assert self.bboxlist is not None, "No bbox sequences exist"
        assert self.masklist is not None, "No mask sequences exist"

        seenObjects = { o for olist in self.objclasslist for o in olist }

        self.objBBMaskSeqDict = dict()
        for objind in list(seenObjects):
            objName = self.thing_classes[objind]
            bbxlist = []
            msklist = []

            for bbxl, mskl, indl in zip(self.bboxlist, self.masklist, self.objclasslist):
                bbxlist.append([bbx for bbx,ind in zip(bbxl,indl) if ind == objind])
                msklist.append([msk for msk,ind in zip(mskl,indl) if ind == objind])

            self.objBBMaskSeqDict[objName] = [bbxlist, msklist]
        
        return len(self.objBBMaskSeqDict)

    def groupObjBBMaskSequence(self,fileglob=None, filelist=None, **kwargs):

        if len(self.imglist) == 0:
            # load images was not called yet
            self.load_images(fileglob=fileglob, filelist=filelist, **kwargs)
        
        if len(self.masklist) == 0:
            # predict images was not called yet
            self.predict_sequence(fileglob=fileglob, filelist=filelist, **kwargs)

        self.__createObjBBMask()
        assert self.objBBMaskSeqDict is not None, "BBox and Mask sequences have not been grouped by objectName"

        self.objBBMaskSeqGrpDict = dict()
        for objName, bblmskl in self.objBBMaskSeqDict.items():
            bbl,mskl = bblmskl
            attGrpBBMsk = []
            for i,bbsmsks in enumerate(zip(bbl,mskl)):
                bbxs,msks = bbsmsks
                if not bbxs: continue
                attGrpBBMsk = self.__assignBBMaskToGroupByDistIndex(attGrpBBMsk,bbxs,msks,i)
    
            self.objBBMaskSeqGrpDict[objName] = attGrpBBMsk
    
    def filter_ObjBBMaskSeq(self,objNameList=None,minCount=10,inPlace=True):
        """
            Performs filtering for minimum group size
            Eventually also for minimum relative object size
        """
        if objNameList is None:
            objNameList = list(self.objBBMaskSeqGrpDict.keys())
        elif not isinstance(objNameList,list):
            objNameList = [objNameList]
        
        assert all([objN in list(self.objBBMaskSeqGrpDict.keys()) for objN in objNameList]), \
            "Invalid list of object names given"
        
        filteredSeq = dict()
        for grpName in objNameList:
            for grp in self.objBBMaskSeqGrpDict[grpName]:
                if len(grp) >= minCount:
                    if not filteredSeq.get(grpName):
                        filteredSeq[grpName] = [grp]
                    else:
                        filteredSeq[grpName].append(grp)
        
        if inPlace:
            self.objBBMaskSeqGrpDict = filteredSeq
            return True
        else:
            return filteredSeq

            
    def get_groupedResults(self, getObjNamesOnly=False, getSpecificObjNames=None):
        """
            Results for sequence prediction, returned as dictionary for objName
            (easily pickelable)
        """
        if getObjNamesOnly:
            return list(self.objBBMaskSeqGrpDict.keys())

        if getSpecificObjNames is not None:
            if not isinstance(getSpecificObjNames,list):
                getSpecificObjNames = [getSpecificObjNames]

            getNames = [ n for n in self.objBBMaskSeqGrpDict.keys() if n in getSpecificObjNames ]
        else:
            getNames = list(self.objBBMaskSeqGrpDict.keys())

        res = { k:v for k,v in self.objBBMaskSeqGrpDict.items() if k in getNames}
        return res


    def combine_MaskSequence(self,objNameList=None, 
                             writeImagesToDirectory=None, 
                             writeMasksToDirectory=None, 
                             cleanDirectory=False, inPlace=True):
        """
            Purpose is to combine all masks at a given time index
            to a single mask. Result is stored 
        """
        if objNameList is None:
            objNameList = list(self.objBBMaskSeqGrpDict.keys())
        elif not isinstance(objNameList,list):
            objNameList = [objNameList]

        assert all([objN in list(self.objBBMaskSeqGrpDict.keys()) for objN in objNameList]), \
            "Invalid list of object names given"

        n_frames = len(self.imglist)

        # write images (which are paired with masks)
        if writeImagesToDirectory is not None:
            imu.writeImagesToDirectory(self.imglist,writeImagesToDirectory,
                                       minPadLength=5,
                                       cleanDirectory=cleanDirectory)

        # combine and write masks
        seqMasks = [ [] for _ in range(n_frames)]
        for objName in objNameList:
            for objgrp in self.objBBMaskSeqGrpDict[objName]:
                for bbx,msk,ind in objgrp:
                    seqMasks[ind].append(msk)

        combinedMasks = [imu.combineMasks(msks) for msks in seqMasks]

        if writeMasksToDirectory is not None:
            imu.writeMasksToDirectory(combinedMasks,writeMasksToDirectory,
                                       minPadLength=5,
                                       cleanDirectory=cleanDirectory)
        
        if inPlace:
            self.combinedMaskList = combinedMasks
            return True
        else:
            return combinedMasks



if __name__ == "__main__":
    pass
