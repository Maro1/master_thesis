from detectron2.projects import point_rend
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2
import numpy as np
import torch
import torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries

# import some common detectron2 utilities

coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project

pred_classes = ['person',
                'bicycle',
                'car',
                'motorcycle',
                'airplane',
                'bus',
                'train',
                'truck',
                'boat',
                'traffic light',
                'fire hydrant',
                'stop sign',
                'parking meter',
                'bench',
                'bird',
                'cat',
                'dog',
                'horse',
                'sheep',
                'cow',
                'elephant',
                'bear',
                'zebra',
                'giraffe',
                'backpack',
                'umbrella',
                'handbag',
                'tie',
                'suitcase',
                'frisbee',
                'skis',
                'snowboard',
                'sports ball',
                'kite',
                'baseball bat',
                'baseball glove',
                'skateboard',
                'surfboard',
                'tennis racket',
                'bottle',
                'wine glass',
                'cup',
                'fork',
                'knife',
                'spoon',
                'bowl',
                'banana',
                'apple',
                'sandwich',
                'orange',
                'broccoli',
                'carrot',
                'hot dog',
                'pizza',
                'donut',
                'cake',
                'chair',
                'couch',
                'potted plant',
                'bed',
                'dining table',
                'toilet',
                'tv',
                'laptop',
                'mouse',
                'remote',
                'keyboard',
                'cell phone',
                'microwave',
                'oven',
                'toaster',
                'sink',
                'refrigerator',
                'book',
                'clock',
                'vase',
                'scissors',
                'teddy bear',
                'hair drier',
                'toothbrush']


class AlphaPredictor:
    """
    Class for predicting segmentation masks using PointRend
    """

    def __init__(self):
        self.cfg = get_cfg()
        # Add PointRend-specific config
        point_rend.add_pointrend_config(self.cfg)
        # Load a config from file
        self.cfg.merge_from_file(
            "detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
        self.cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"

        self.predictor = DefaultPredictor(self.cfg)

    def find_mask(self, im, category):

        try:
            outputs = self.predictor(im)
        except:
            return None

        if len(outputs['instances'].scores) == 0:
            return None

        # If category is in the predicted classes, use this mask
        classes = outputs['instances'].pred_classes.cpu().numpy()
        if pred_classes.index(category) in classes:
            mask = outputs['instances'].pred_masks[np.where(
                classes == pred_classes.index(category))[0]]
            return mask.cpu().numpy()

        # Otherwise use mask with highest score
        max_score = torch.max(outputs['instances'].scores)
        max_idx = torch.argmax(outputs['instances'].scores)
        mask = outputs['instances'].pred_masks[max_idx.item()]
        return mask.cpu().numpy()
