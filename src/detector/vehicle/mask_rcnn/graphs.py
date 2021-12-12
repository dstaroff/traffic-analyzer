import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Input,
    Lambda,
    MaxPooling2D,
    Reshape,
    TimeDistributed,
    ZeroPadding2D,
    )
from tensorflow.keras.models import Model

from .blocks import (
    conv_block,
    identity_block,
    )
from .layers.roi_align import PyramidROIAlign


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """
    Build a ResNet graph.
    - architecture: Can be resnet50 or resnet101
    - stage5: Boolean. If False, stage5 of the network is not created
    - train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]

    # Stage 1
    x = ZeroPadding2D((3, 3))(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)
    c1 = x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    c2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    c3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    c4 = x

    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        c5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        c5 = None

    return [c1, c2, c3, c4, c5]


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically, 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
    """
    # Shared convolutional base of the RPN
    shared = Conv2D(
            512, (3, 3), padding='same', activation='relu',
            strides=anchor_stride,
            name='rpn_conv_shared'
            )(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = Conv2D(
            2 * anchors_per_location, (1, 1),
            activation='linear', name='rpn_class_raw'
            )(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = Lambda(
            lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2])
            )(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = Conv2D(
            anchors_per_location * 4, (1, 1),
            activation='linear', name='rpn_bbox_pred'
            )(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph, so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically, 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = Input(
            shape=[None, None, depth],
            name="input_rpn_feature_map"
            )
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return Model([input_feature_map], outputs, name="rpn_model")


def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024
                         ):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (metadata)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign(
            [pool_size, pool_size],
            name="roi_align_classifier"
            )([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = TimeDistributed(
            Conv2D(fc_layers_size, (pool_size, pool_size)),
            name="mrcnn_class_conv1"
            )(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(
            Conv2D(fc_layers_size, (1, 1)),
            name="mrcnn_class_conv2"
            )(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    shared = Lambda(
            lambda y: tf.squeeze(tf.squeeze(y, 3), 2),
            name="pool_squeeze"
            )(x)

    # Classifier head
    mrcnn_class_logits = TimeDistributed(
            Dense(num_classes),
            name='mrcnn_class_logits'
            )(shared)
    mrcnn_probs = TimeDistributed(
            Activation("softmax"),
            name="mrcnn_class"
            )(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = TimeDistributed(
            Dense(num_classes * 4, activation='linear'),
            name='mrcnn_bbox_fc'
            )(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]

    mrcnn_bbox = Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True
                         ):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (metadata)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign(
            [pool_size, pool_size],
            name="roi_align_mask"
            )([rois, image_meta] + feature_maps)

    # Conv layers
    x = TimeDistributed(
            Conv2D(256, (3, 3), padding="same"),
            name="mrcnn_mask_conv1"
            )(x)
    x = TimeDistributed(
            BatchNormalization(),
            name='mrcnn_mask_bn1'
            )(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(
            Conv2D(256, (3, 3), padding="same"),
            name="mrcnn_mask_conv2"
            )(x)
    x = TimeDistributed(
            BatchNormalization(),
            name='mrcnn_mask_bn2'
            )(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(
            Conv2D(256, (3, 3), padding="same"),
            name="mrcnn_mask_conv3"
            )(x)
    x = TimeDistributed(
            BatchNormalization(),
            name='mrcnn_mask_bn3'
            )(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(
            Conv2D(256, (3, 3), padding="same"),
            name="mrcnn_mask_conv4"
            )(x)
    x = TimeDistributed(
            BatchNormalization(),
            name='mrcnn_mask_bn4'
            )(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(
            Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
            name="mrcnn_mask_deconv"
            )(x)
    x = TimeDistributed(
            Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
            name="mrcnn_mask"
            )(x)
    return x
