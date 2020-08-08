import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from SSD.bounding_box_utils import convert_coordinates


class AnchorBoxes(layers.Layer):
    """
    A Keras layer to create an output tensor containing anchor box coordinates and variances based on the input tensor
    and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of the input tensor. The number
    of anchor boxes created per unit depends on the arguments 'aspect_ratios' and 'two_boxes_for_ar1', in the default case
    it is 4. The boxes are parameterized by the coordinate tuple (xmin, xmax, ymin, ymax).

    Since the model is predicting offsets to the anchor boxes (rather than predicting absolute box coordinate directly),
    the anchor box coordinate in order to construct the final prediction boxes from the predicted offsets. If the model's
    output tensor did not contain the anchor box coordinates, the necessary information to convert the predicted offsets
    back to absolute coordinates would be missing in the model output. The reason why it is necessary to predict offsets
    to the anchor boxes rather than to predict absolute box coordinates directly

    Input shape:
        4D tensor of shape: (batch, height, width, channels)
    Output shape:
        5D tensor of shape (batch, height, width, n_boxes, 8). The last axis contains the four anchor box coordinates and
        the four variance for each box
    """

    def __init__(self, img_height, img_width, this_scale, next_scale, aspect_ratios=None, two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None, clip_boxes=False, variances=None, coords='centroids', normalize_coords=False,
                 **kwargs):
        """
        :param img_height: (int) The height of the input images.
        :param img_width: (int) The width of the input images
        :param this_scale: (float) A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                            as a fraction of the shorter side of the input image
        :param next_scale: (float): A float in [0, 1], the next larger scaling factor. Only relevant if 'self.two_boxes_for_ar1=True'
        :param aspect_ratios: (list, optional): The list of aspect ratios for which default boxes are to be generated for this layer
        :param two_boxes_for_ar1: (bool, optional): Only relevant if 'aspect ratio' contains 1. If 'true', two default boxes
                                    will be generated using the scaling factor for the respective layer, the second one will
                                    be generated using geometric mean of said scaling factor and next bigger scaling factor
        :param clip_boxes: (bool, optional): If True, clips the anchor box coordinates to stay within image boundaries
        :param variances: A list of 4 floats > 0. The anchor box offset for each coordinate will be divided by its respective
        variance value
        :param coords:
        :param normalize_coords:
        :param kwargs:
        """
        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError(
                "`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(
                    this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be passed, but {} values were received.".format(len(variances)))

        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.normalize_coords = normalize_coords
        self.coords = coords

        if aspect_ratios is None:
            self.aspect_ratios = [0.5, 1.0, 2.0]
        else:
            self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        if variances is None:
            self.variances = [0.1, 0.1, 0.2, 0.2]
        else:
            self.variances = variances

        # compute the number of boxes per cell
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [layers.InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        """
        Returns an anchor box tensor based on the shape of the input tensor
        :param x: (batch, height, width, channels) The input for this layer must be the output of the localization predictor layer
        :param mask:
        :return:
        """

        # compute box width and height for each aspect ratio
        size = min(self.img_height, self.img_width)

        # compute the box widths and heights for all aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if ar == 1:
                # compute the regular anchor box for aspect ratio 1
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # compute one slightly larger version using the geometric mean of this scale value and the next
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))

        wh_list = np.array(wh_list)

        batch_size, feature_map_height, feature_map_width, feature_map_channels = x.get_shape().as_list()

        # compute the grid of box center points. They are identical for all aspect ratio
        if self.this_steps is None:
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps

        # compute the offsets, i.e what pixel values the first anchor box center point will be from the top and from the
        # left of the image
        if self.this_offsets is None:
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets

        # Compute the grid of anchor box center points
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, axis=-1)
        cy_grid = np.expand_dims(cy_grid, axis=-1)

        # create a 4D tensor tensor template of shape (feature_map_height, feature_map_width, n_boxes, 4) where the last
        # dimension will contain (cx, cy, w, h)
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]

        # convert (cx, cy, w, h) to (xmin, xmax, ymin, ymax)
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If clip boxes is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords

            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        # if normalize coords is enabled, normalize the coordinates to be within [0, 1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        if self.coords == 'centroids':
            # convert (xmin, ymin, xmax, ymax) back to (cx, cy, w, h)
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids')
        elif self.coords == 'minmax':
            # convert (xmin, ymin, xmax, ymax) to (xmin, xmax, ymin, ymax)
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax')

        # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)  # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances  # Long live broadcasting

        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape

        return batch_size, feature_map_height, feature_map_width, self.n_boxes, 8

    @property
    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

















