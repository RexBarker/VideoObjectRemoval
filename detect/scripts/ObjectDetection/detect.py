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

# plotting utilities
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# module specific library
import ObjectDetection.imutils as imu

# ---------------------------------------------------------------------
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
        self.DEVICE = None

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
        self.DEVICE = self.predictor.model.device.type + str(self.predictor.model.device.index)

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
        scores = list(outputs['instances'].scores.cpu().numpy()) 
        objects = [self.thing_classes[c] in selObjectNames for c in classes]

        self.outputs = outputs
        self.masks = [ outputs['instances'].pred_masks[i].cpu().numpy() for i,o in enumerate(objects) if o ]
        self.bboxes = [ imu.bboxToList(outputs['instances'].pred_boxes[i]) for i,o in enumerate(objects) if o ]
        self.scores = [ scores[i] for i,o in enumerate(objects) if o ]

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

# ---------------------------------------------------------------------
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
    

# ---------------------------------------------------------------------
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
        self.orginalSequenceMap = None
        self.MPEGconfig = {
            'fps': 50,
            'metadata': {'artist': "appuser"},
            'bitrate' : 1800
        }

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
        
        self.orginalSequenceMap = \
            {objn:{ i:None for i in range(len(self.objBBMaskSeqGrpDict[objn]))} \
                for objn in self.objBBMaskSeqGrpDict.keys() }

        filteredSeq = dict()
        for grpName in objNameList:
            inew = 0
            for iold, grp in enumerate(self.objBBMaskSeqGrpDict[grpName]):
                if len(grp) >= minCount:
                    if not filteredSeq.get(grpName):
                        filteredSeq[grpName] = [grp]
                    else:
                        filteredSeq[grpName].append(grp)

                    self.orginalSequenceMap[grpName][iold] = inew
                    inew += 1
        
        if inPlace:
            self.objBBMaskSeqGrpDict = filteredSeq
            return True
        else:
            return filteredSeq


    def fill_ObjBBMaskSequence(self, specificObjectNameInstances=None):
        """
            Purpose: fill in missing masks of a specific object sequence
            masks are stretched to interpolated bbox region
            'specificObjectNameInstances' = { {'objectName', [0, 2 , ..]}
            Examples:
                'specificObjectNameInstances' = { 'person', [0, 2 ]}  # return person objects, instance 0 and 2
                'specificObjectNameInstances' = { 'person', None }  # return 'person' objects, perform for all instances
                'specificObjectNameInstances' = None  # perform for all object names, for all instances 
            Note: the indices of the 'specificObjectNameInstances' refer to the orginal order before filtering
        """
        def maximizeBBoxCoordinates(bbxlist):
            # provides maximum scope of set of bbox inputs
            if isinstance(bbxlist[0],(float,int)):  
                return bbxlist  # single bounding box passed, first index is x0
    
            # multiple bbx
            minx,miny,maxx,maxy = bbxlist[0]
            
            if len(bbxlist) > 1:
                for x0,y0,x1,y1 in bbxlist[1:]:
                    minx = min(minx,x0)
                    maxx = max(maxx,x1)
                    miny = min(miny,y0)
                    maxy = max(maxy,y1)
            
            return [minx,miny,maxx,maxy] 

        allObjNameIndices = { objn: [range(len(obji))] for objn,obji in self.objBBMaskSeqGrpDict.items() }
        allObjNames = list(self.objBBMaskSeqGrpDict.keys())

        if specificObjectNameInstances is None:   # all objects, all instances
            objNameIndices = allObjNameIndices
            objIndexMap = { objn: {i:i for i in range(len(objl))} \
                            for objn,objl in self.objBBMaskSeqGrpDict.items() }
        else:
            assert isinstance(specificObjectNameInstances,dict), \
                "Expected a dictionary object for 'specificeeObjectNameInstances'"

            assert all([o in allObjNames for o in specificObjectNameInstances.keys()]), \
                "Object name specified which are not in predicted list"
            
            if self.orginalSequenceMap is None:
                assert all([max(obji) < len(self.objBBMaskSeqGrpDict[objn]) for objn,obji in specificObjectNameInstances.items() ]), \
                    "Specified objectName index exceeded number of known instances of objectName from detection"

                objIndexMap = { objn: {i:i for i in range(len(objl))} \
                                for objn,objl in self.objBBMaskSeqGrpDict.items() }
            else:
                for objn,objl in specificObjectNameInstances.items():
                    # check that the original object index was mapped after filtering
                    assert all([self.orginalSequenceMap[objn][i] is not None for i in objl]), \
                        f"Specified object '{objn}' index list was invalid due to missing index:[{objl}]"

                objIndexMap = {objn: {i:self.orginalSequenceMap[objn][i] \
                               for i in objl} for objn,objl in specificObjectNameInstances.items()}
        

        for objn, objindices in specificObjectNameInstances.items():
            for objiold in objindices:
                objinew = objIndexMap[objn][objiold]
                rseq = self.objBBMaskSeqGrpDict[objn][objinew]

                mskseq = [[] for _ in range(len(self.imglist))]
                bbxseq = [[] for _ in range(len(self.imglist))]

                for bbxs, msks, ind in rseq:
                    bbx = maximizeBBoxCoordinates(bbxs)
                    msk = imu.combineMasks(msks)
                    bbxseq[ind]=bbx
                    mskseq[ind]=msk
                
                # build dataframe of known bbox coordinates 
                targetBBDF = pd.DataFrame(bbxseq,columns=['x0','y0','x1','y1'])

                # determine missing indices
                missedIndices = [index for index, row in targetBBDF.iterrows() if row.isnull().any()]

                # extrapolate missing bbox coordinates
                targetBBDF = targetBBDF.interpolate(limit_direction='both', kind='linear')

                # output bboxes, resequenced
                bbxseq = [[r.x0,r.y0,r.x1,r.y1] for _,r,*_ in targetBBDF.iterrows()] 

                # create missing masks by stretching or shrinking known masks from i-1 
                # (relies on prior prediction of missing mask, for sequential missing masks)
                for i in missedIndices:
                    lasti = i-1   # can't have i=0 missing, otherwise there's a corrupt system
                    lastmsk = mskseq[lasti]
                    
                    # masks which were not good are found here
                    x0o,y0o,x1o,y1o = [round(v) for v in targetBBDF.iloc[lasti]]
                    x0r,y0r,x1r,y1r = [round(v) for v in targetBBDF.iloc[i]]
                    
                    wr = x1r - x0r
                    hr = y1r - y0r
                    
                    msko = mskseq[lasti]*1.0
                    mskr = np.zeros_like(msko)
                    
                    submsko = msko[y0o:y1o,x0o:x1o]
                    submskr = cv2.resize(submsko,(wr,hr))
                    
                    mskr[y0r:y1r,x0r:x1r] = submskr
                    mskseq[i] = mskr > 0.0   # returns np.array(dtype=np.bool)
         
                # recollate into original class object, with the new definitions
                outrseq = [ [bbxmsk[0],bbxmsk[1], ind] for ind,bbxmsk in enumerate(zip(bbxseq, mskseq))]
                self.objBBMaskSeqGrpDict[objn][objinew] = outrseq

        return True        


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


    def dilateErode_MaskSequence(self,masklist=None,
                                      actionList=['dilate'], 
                                      kernelShape='rect',
                                      maskHalfWidth=4,
                                      inPlace=True):
        """
            Purpose: to dilate or erode mask, where the mask list is
            either the exisitng mask list or one passed. The mask list must
            be a flat list of single masks for one frame index (all masks must have been combined)
        """
        if masklist is None:
            if self.combinedMaskList is None:
                raise Exception("No masks were given for dilateErode operation")
            else:
                masklist = self.combinedMaskList
        
        maskListOut = []
        for msk in masklist:
            maskListOut.append(imu.dilateErodeMask(msk, actionList=actionList,
                                                        kernelShape=kernelShape,
                                                        maskHalfWidth=maskHalfWidth)) 
        if inPlace:
            self.combinedMaskList = maskListOut 
            return True
        else:
            return maskListOut 


    def combine_MaskSequence(self,objNameList=None, 
                              inPlace=True):
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

        # combine and write masks
        seqMasks = [ [] for _ in range(n_frames)]
        for objName in objNameList:
            for objgrp in self.objBBMaskSeqGrpDict[objName]:
                for bbx,msk,ind in objgrp:
                    seqMasks[ind].append(msk)

        combinedMasks = [imu.combineMasks(msks) for msks in seqMasks]
        if inPlace:
            self.combinedMaskList = combinedMasks
            return True
        else:
            return combinedMasks


    def write_ImageMaskSequence(self,imagelist=None,
                                masklist=None,
                                writeMasksToDirectory=None, 
                                writeImagesToDirectory=None,
                                cleanDirectory=False):

        if imagelist is None:
            if self.imglist is not None:
                imagelist = self.imglist  

        if masklist is None:
            if self.combinedMaskList is not None:
                masklist = self.combinedMaskList

        # write images (which are paired with masks)
        if (writeImagesToDirectory is not None) and (imagelist is not None):
            imu.writeImagesToDirectory(imagelist,writeImagesToDirectory,
                                       minPadLength=5,
                                       cleanDirectory=cleanDirectory)

        # write masks (which are paired with masks)
        if (writeMasksToDirectory is not None) and (masklist is not None) :
            imu.writeMasksToDirectory(masklist,writeMasksToDirectory,
                                       minPadLength=5,
                                       cleanDirectory=cleanDirectory)
        
        return True


    def create_animationObject(self,
                            getSpecificObjNames=None,
                            framerange=None,
                            useMasks=True,
                            toHTML=False,
                            MPEGfile=None,
                            MPEGconfig=None,
                            figsize=(10,10),
                            interval=30,
                            repeat_delay=1000):
        """
            Purpose: produce an animation object of the masked frames 
            returns an animation object to be rendered with HTML()
        """
        if getSpecificObjNames is not None:
            if not isinstance(getSpecificObjNames,list):
                getSpecificObjNames = [getSpecificObjNames]

            getNames = [ n for n in self.objBBMaskSeqGrpDict.keys() if n in getSpecificObjNames ]
        else:
            getNames = list(self.objBBMaskSeqGrpDict.keys())
        
        if framerange is None:
            framemin,framemax = 0, len(self.imglist)-1
        else:
            framemin,framemax = framerange

        fig = plt.figure(figsize=figsize)
        plt.axis('off')

        # combine and write masks
        if useMasks:
            if self.combinedMaskList is None:
                seqMasks = [ [] for _ in range(framemin,framemax+1)]
                for objName in getNames:
                    for objgrp in self.objBBMaskSeqGrpDict[objName]:
                        for bbx,msk,ind in objgrp:
                            seqMasks[ind].append(msk)
            else:
                seqMasks = self.combinedMaskList

        outims = []
        for i,im in enumerate(self.imglist):
            if useMasks:
                msks = seqMasks[i] 
                if isinstance(msks,list):
                    for gi,msk in enumerate(msks):
                        im = imu.maskImage(im,msk,self.thing_colors[gi])
                else:
                    im = imu.maskImage(im,msks,self.thing_colors[0])

                
            im = im[:,:,::-1]  # convert from BGR to RGB
            renderplt = plt.imshow(im,animated=True)
            outims.append([renderplt])
        
        ani = animation.ArtistAnimation(fig, outims, interval=interval, blit=True,
                                repeat_delay=repeat_delay)

        if MPEGfile is not None:
            # expecting path to write file
            assert os.path.isdir(os.path.dirname(MPEGfile)), f"Could not write to path {os.path.dirname(MPEGfile)}"

            if MPEGconfig is None:
                MPEGconfig = self.MPEGconfig

            mpegWriter = animation.writers['ffmpeg']
            writer = mpegWriter(**MPEGconfig)
            ani.save(MPEGfile,writer=writer)

        # return html object, or just animation object  
        return ani.to_html5_video() if toHTML else ani


if __name__ == "__main__":
    pass
