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
        elif isinstance(img,np.array):
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
            outim = imu.mask_image(outim,msk,color)
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

    def predict_sequence(self,fileglob=None, filespec=None, **kwargs):
        if self.imglist is None:
            # load images was not called yet
            self.load_images(fileglob=fileglob, filespec=filespec, **kwargs)

        for im in self.imglist:
            self.predict(im)  
            self.masklist.append(self.masks)
            self.bboxlist.append(self.bboxes)
            self.objclasslist.append(self.selClassList)
        
        return len(self.masklist) 

    def get_results(self,getImage=True, getMasks=True, getBBoxes=True, getClasses=True):
        res = dict()
        if getImage: res['im'] = self.imglist
        if getMasks: res['masks'] = self.masklist
        if getBBoxes: res['bboxes'] = self.bboxlist
        if getClasses: res['classes'] = self.objclasslist

        return res 


if __name__ == "__main__":
    pass
