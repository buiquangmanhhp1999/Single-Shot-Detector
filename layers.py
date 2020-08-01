from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Activation, Lambda, Input, Reshape, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import numpy as np
from SSD.AnchorBoxes import AnchorBoxes
from SSD.DecodeDetections import DecodeDetections
from SSD.config import var, aspect_ratios_per_layer, swap_channels, li_steps, subtract_mean, img_width, \
    img_height, img_channels, classes


def vgg16(inputs, l2_reg=0.0005):
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_1')(inputs)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    return pool5, conv4_3


def identity_layer(tensor):
    return tensor


def input_mean_normalization(tensor):
    return tensor - np.array(subtract_mean)


def input_stddev_normalization(tensor, divide_by_stddev):
    return tensor / np.array(divide_by_stddev)


def input_channel_swap(tensor):
    if len(swap_channels) == 3:
        return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]],
                       axis=-1)
    elif len(swap_channels) == 4:
        return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                        tensor[..., swap_channels[3]]], axis=-1)


def ssd_300(mode='training', l2_reg=0.0005, min_scale=None, max_scale=None, scales=None,
            aspect_ratios_global=None,
            two_boxes_for_ar1=True, offsets=None, clip_boxes=False, coords='centroids', normalize_coords=True,
            divide_by_stddev=None,
            confidence_thresh=0.01, iou_threshold=0.45, top_k=200, nms_max_output_size=400):
    """
    Build a Keras model with SSD300 architecture
    :param mode: Have 3 option: 'training', 'reference', 'inference_fast'
    :param l2_reg: L2 regularization rate. Applies to all convolutional layers
    :param min_scale: (float, optional) The smallest scaling factor for the size of the anchor boxes as a fraction of the shorter
            side of the input images
    :param max_scale: (float, optional) The largest scaling factor for the size of the anchor boxes as a fraction of the shorted side
            of input images
    :param scales: (list, optional): A list of floats containing scaling factors per convolutional predictor layer. This list
            must be one element longer than the number of predictor layers. The first 'k' elements are the scaling factors
            for the 'k' predictor layers, while the last element is used for the second for aspect ratio 1 in the last predictor
            layers if 'two_boxes_for_ar1' is 'True'. This additional last scaling factor must be passed either way, even if
            it is not being used. If a list is passed, this argument overrides 'min_scale' and 'max_scale'. All scaling
            factors must be greater than zero
    :param aspect_ratios_global:(list, optional): The list of aspect ratios for which anchor boxes are to be generated. This
            list is valid for all prediction layers
    :param two_boxes_for_ar1:(bool, optional): Only relevant for aspect ratio list contain 1. Will be ignored otherwise.
            If 'True', two anchor boxes will be generated for aspect ratio 1. The first will be generated using the scaling factor for
            the scaling factor for the respective layer, the second one will be generated using geometric mean of said scaling
            factor and next bigger scaling factor
    :param offsets: `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
    :param clip_boxes: If `True`, clips the anchor box coordinates to stay within image boundaries.
    :param coords: (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
    :param normalize_coords: Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
    :param divide_by_stddev: `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
    :param confidence_thresh: (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            threshold stage.
    :param iou_threshold:
    :param top_k: (int, optional)The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
    :param nms_max_output_size: he maximal number of predictions that will be left over after the NMS stage.
    :return: model keras SSD 300
    """
    # the number of predictor conv layers in the network is 6 for the original SSD300
    n_predictor_layers = 6
    n_classes = classes + 1  # account for the background class

    # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
    if scales is None:
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)
    else:
        scales = scales

    variances = np.array(var)
    # set the aspect ratio for each predictor layer
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # + 1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:
        # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)

        n_boxes = [n_boxes] * n_predictor_layers

    if li_steps is None:
        steps = [None] * n_predictor_layers
    else:
        steps = li_steps

    if offsets is None:
        offsets = [None] * n_predictor_layers

    # Build the network
    x = Input(shape=(img_height, img_width, img_channels))

    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)

    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x)

    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='')

    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(
            x1)

    base_network, conv4_3 = vgg16(inputs=x1, l2_reg=l2_reg)

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc6')(base_network)

    fc7 = Conv2D(1024, kernel_size=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = Lambda(lambda y: K.l2_normalize(y, axis=3))(conv4_3)

    # Build the convolutional predictor layers on top of the base network
    # Predict 'n_classes' confidence values for each box, the confidence predictor have depth 'n_boxes * n_classes'
    # Output shape of the confidence layers: '(batch, height, width, n_boxes * n_classes)
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)

    # Predict 4 box coordinate for each box, hence the localization predictors have depth 'n_boxes * 4'
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)

    # Generate the anchor boxes: (batch, height, width, n_boxes, 8)
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_prior_box = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                              aspect_ratios=aspect_ratios[0],
                                              two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                              this_offsets=offsets[0], clip_boxes=clip_boxes,
                                              variances=variances, coords=coords, normalize_coords=normalize_coords,
                                              name='conv4_3_norm_mbox_prior_box')(conv4_3_norm_mbox_loc)
    fc7_mbox_prior_box = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                     aspect_ratios=aspect_ratios[1],
                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                     clip_boxes=clip_boxes,
                                     variances=variances, coords=coords, normalize_coords=normalize_coords,
                                     name='fc7_mbox_prior_box')(fc7_mbox_loc)
    conv6_2_mbox_prior_box = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                         aspect_ratios=aspect_ratios[2],
                                         two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                         this_offsets=offsets[2], clip_boxes=clip_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv6_2_mbox_prior_box')(conv6_2_mbox_loc)
    conv7_2_mbox_prior_box = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                         aspect_ratios=aspect_ratios[3],
                                         two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                         this_offsets=offsets[3], clip_boxes=clip_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv7_2_mbox_prior_box')(conv7_2_mbox_loc)
    conv8_2_mbox_prior_box = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                         aspect_ratios=aspect_ratios[4],
                                         two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                         this_offsets=offsets[4], clip_boxes=clip_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv8_2_mbox_prior_box')(conv8_2_mbox_loc)
    conv9_2_mbox_prior_box = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                         aspect_ratios=aspect_ratios[5],
                                         two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                         this_offsets=offsets[5], clip_boxes=clip_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords,
                                         name='conv9_2_mbox_prior_box')(conv9_2_mbox_loc)

    # Reshape the class prediction from (batch, height, width, n_boxes * n_classes) to (batch, height * width * n_boxes, n_classes)
    # we want the class isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
        conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_mbox_conf_reshape')(conv9_2_mbox_conf)

    # Reshape the box prediction from (batch, height, width, n_boxes * 4) to (batch, height * width * n_boxes, 4)
    # we want the four box coordinates isolated in the last axis to compute the smooth L1 Loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

    # Reshape the anchor box tensors from (batch, height, width, n_boxes, 8) to (batch, height * width * n_boxes, 8)
    conv4_3_norm_mbox_prior_box_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_prior_box_reshape')(
        conv4_3_norm_mbox_prior_box)
    fc7_mbox_prior_box_reshape = Reshape((-1, 8), name='fc7_mbox_prior_box_reshape')(fc7_mbox_prior_box)
    conv6_2_mbox_prior_box_reshape = Reshape((-1, 8), name='conv6_mbox_prior_box_reshape')(conv6_2_mbox_prior_box)
    conv7_2_mbox_prior_box_reshape = Reshape((-1, 8), name='conv7_2_mbox_prior_box_reshape')(conv7_2_mbox_prior_box)
    conv8_2_mbox_prior_box_reshape = Reshape((-1, 8), name='conv8_2_mbox_prior_box_reshape')(conv8_2_mbox_prior_box)
    conv9_2_mbox_prior_box_reshape = Reshape((-1, 8), name='conv9_2_mbox_prior_box_reshape')(conv9_2_mbox_prior_box)

    # Concatenate the prediction from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape, fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape, conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape, conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape, fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape, conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape, conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_prior_box`: (batch, n_boxes_total, 8)
    mbox_prior_box = Concatenate(axis=1, name='mbox_prior_box')([conv4_3_norm_mbox_prior_box_reshape,
                                                                 fc7_mbox_prior_box_reshape,
                                                                 conv6_2_mbox_prior_box_reshape,
                                                                 conv7_2_mbox_prior_box_reshape,
                                                                 conv8_2_mbox_prior_box_reshape,
                                                                 conv9_2_mbox_prior_box_reshape])

    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_prior_box])

    if mode == 'train':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_prediction = DecodeDetections(confidence_thresh, iou_threshold, top_k, nms_max_output_size,
                                              coords, normalize_coords, img_height, img_width,
                                              name='decoded_prediction')(predictions)
        model = Model(inputs=x, outputs=decoded_prediction)

    return model
