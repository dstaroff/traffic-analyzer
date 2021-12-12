import numpy as np

from src.utils.const import MAX_MASKRCNN_WIDTH


class Config(object):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Supported values are: resnet50, resnet101.
    BACKBONE = 'resnet101'
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 1
    RPN_NMS_THRESHOLD = 0.7
    PRE_NMS_LIMIT = 6000
    POST_NMS_ROIS_INFERENCE = 1000
    IMAGE_RESIZE_MODE = 'none'  # 'square'
    IMAGE_MIN_DIM = MAX_MASKRCNN_WIDTH
    IMAGE_MAX_DIM = MAX_MASKRCNN_WIDTH
    IMAGE_MIN_SCALE = 0
    IMAGE_CHANNEL_COUNT = 3
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.65
    DETECTION_NMS_THRESHOLD = 0.3

    TRAIN_BN = False  # Defaulting to False since batch size is often small

    def __init__(self, num_classes: int):
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        self.IMAGE_SHAPE = np.array(
                [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                 self.IMAGE_CHANNEL_COUNT]
                )
        self.NUM_CLASSES = num_classes
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
