from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Activation, Lambda, Input, Reshape, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2


def vgg16(inputs, l2_reg=0.0005):
    nb_filters = [64, 128, 256, 512, 512]

    x = inputs
    for i, nb_filter in enumerate(nb_filters):
        x = Conv2D(nb_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv1_' + str(i + 1))(x)
        x = Conv2D(nb_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv2_' + str(i + 1))(x)
        if i == 4:
            x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool' + str(i + 1))(x)
        else:
            x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool' + str(i + 1))(x)

    return x


def ssd_300(image_size, n_classes, mode='training', l2_reg=0.0005, min_scale=None, max_scale=None, scales=None, aspect_ratios_global=None,
            two_boxes_for_ar1=True, offsets=None, clip_boxes=False, coords='centroids', normalize_coords=True, divide_by_stddev=None,
            confidence_thresh=0.01, iou_threshold=0.45, top_k=200, nms_max_output_size=400):
    """
    Build a Keras model with SSD300 architecture
    :param image_size: The input image size in the format (height, width, channels)
    :param n_classes: The number of positive classes
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
    aspect_ratios_per_layer = [[1.0, 2.0, 0.5],
                               [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                               [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                               [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                               [1.0, 2.0, 0.5],
                               [1.0, 2.0, 0.5]]
    steps = [8, 16, 32, 64, 100, 300]
    variances = [0.1, 0.1, 0.2, 0.2]
    subtract_mean = [123, 117, 104]
    swap_channels = [2, 1, 0]

