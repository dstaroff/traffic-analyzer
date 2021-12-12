import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

APP_NAME = 'traffic-analyzer'
APP_DISPLAY_NAME = 'Traffic analyzer'

PREF_FPS = 30
PREF_VIDEO_W = 512

FIELD_OPACITY = 0.3

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

DEFAULT_DISTANCE = 10

DATA_PATH = os.path.abspath(os.path.join(PROJECT_PATH, 'data'))
MODEL_WEIGHTS_PATH = os.path.join(DATA_PATH, "weights.h5")

CLASS_NAMES_ALL = (
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    )

CLASS_NAMES_USED = ('bicycle', 'car', 'motorcycle', 'bus', 'truck')

MAX_MASKRCNN_WIDTH = 512

CAR_MASK_COLOR = COLOR_RED
