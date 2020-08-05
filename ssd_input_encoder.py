from SSD.bounding_box_utils import iou, convert_coordinates
import numpy as np
from SSD.config import img_height, img_width, classes, aspect_ratios_per_layer, li_steps, offsets, var


class SSDInputEncoder:
    """
    Transforms ground truth labels for object detection in images(2D bounding box coordinates and class labels) to the
    format required for training an SSD model

    In the process of encoding the ground truth labels, a template of anchor boxes is being built, which are subsequently matched
    to the ground truth boxes via an intersection-over-union threshold criterion
    """
    def __init__(self, predictor_sizes, min_scale=0.1, max_scale=0.9, scales=None, clip_boxes=False, two_boxes_for_ar1=True, matching_type='multi',
                 pos_iou_threshold=0.5, neg_iou_limit=0.3, border_pixels='half', coords='centroids', normalize_coords=True,
                 background_id=0):
        """
        :param predictor_sizes:  (list) A list of int-tuples of the format (height, width) containing the output heights and widths
                                of the convolutional predictor layers
        :param min_scale: (float, optional) The smallest scaling factor fot the size of the anchor boxes as a fraction of the shorter side
                of the input images. Note the you should set the scaling factors such that the resulting anchor box sizes correspond
                to the sizes of the objects you are trying detect. Must be > 0
        :param max_scale: (float, optional) The largest scaling factor for the size of the anchor boxes as a fraction of the
                shorter side of the input images. All scaling factor between the smallest and the largest will be linearly interpolated.
                Note that the second to last of the linearly interpolated scaling factors will actually be the scaling factor for the last
                predictor layer, while the last scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
                if 'two_boxes_for_ar1' is True. Note that you should set the scaling factors such that th resulting anchor box
                sizes correspond to the sizes of the objects you are trying to detect.
        :param scales: (list, optional): A list of floats > 0 containing scaling factors per convolutional predictor layer.
                This list must be one element longer than the number of predictor layers. The first 'k' elements are the scaling
                factor for the k predictor layers, while the last element is used for the second box for aspect ratio 1 in the last
                predictor layer if 'two_boxes_for_ar1' is 'True'
        :param two_boxes_for_ar1: (bool, optional) Only relevant for aspect ratios lists that contain 1. Will be ignored otherwise
                                    If 'True', two scaling factor for the respective layer, the second one will be generated using
                                    geometric mean of said scaling factor and next bigger scaling factor
        :param matching_type:(str, optional) Can be either 'multi' or 'bipartite'. In 'bipartite' mode, each ground truth box will
                be matched only to the one anchor box with the highest IoU overlap. In 'multi' mode, in addition to the aforementioned
                bipartite matching, all anchor boxes with an IoU overlap greater than or equal to the 'pos_iou_threshold' will be
                matched to a given ground truth box.
        :param pos_iou_threshold: (float, optional) The intersection-over-union similarity threshold that must be met in order to match
                a given ground truth box to a given anchor box
        :param neg_iou_limit: (float, optional) The maximum allowed intersection-over-union similarity of an anchor box with
                any ground truth box to be labeled a negative box (background). If an anchor box is neither a positive, nor a negative box
                it will be ignored during training
        :param border_pixels:(str, optional) How to treat the border pixels of the bounding boxes. Can be 'include', 'exclude',
                or 'half'. If 'include', the border pixels of the bounding boxes belong to the boxes. If 'exclude', the border pixels
                do not belong to the boxes. If 'half', then one of each of two horizontal and vertical borders belong to the boxes,
                but not the other.
        :param coords:(str, optional) The box coordinate format to be used internally by the model
        :param normalize_coords: (bool, optional) If 'True', the encoder uses relative instead of absolute coordinates. This means
                instead of using absolute target coordinates, the encoder will scale all coordinates to be within [0, 1]. Thos
                way learning become independent of the input image size
        :param background_id:(int, optional): Determines which class ID is for the background class
        """
        self.clip_boxes = clip_boxes
        predictor_sizes = np.array(predictor_sizes)
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if len(scales) != predictor_sizes.shape[0] + 1:  # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}".format(len(scales), len(predictor_sizes) + 1))
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError("All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
        else:  # If no list of scales was passed, we need to make sure that `min_scale` and `max_scale` are valid values.
            if not 0 < min_scale <= max_scale:
                raise ValueError("It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(min_scale, max_scale))

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = classes
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale

        # If scales is None, compute the scaling factors by linearly interpolating between min_scale and max_scale
        if scales is None:
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes) + 1)
        else:   # if a list of scales is given explicitly, we'll use that instead of computing it from min_scale and max_scale
            self.scales = scales

        self.aspect_ratios = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.offsets = offsets
        self.clip_boxes = clip_boxes
        self.variances = var
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coord = coords
        self.normalize_coords = normalize_coords
        self.background_id = background_id

        # Compute the number of boxes per spatial location for each predictor layer.
        # For example, if a predictor layer has three different aspect ratios. [1.0, 0.5, 2.0] and is supposed to predict two
        # boxes of slightly different size fro aspect ratio 1.0, then that predictor layer predicts a total of four boxes
        # at every spatial location across the feature map
        if aspect_ratios_per_layer is None:
            raise ValueError('Missing aspect ratio per layer value')

        self.n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:
            if (1 in aspect_ratios) & two_boxes_for_ar1:
                self.n_boxes.append(len(aspect_ratios) + 1)
            else:
                self.n_boxes.append(len(aspect_ratios))

        # Compute the anchor boxes for each predictor layer. We only have to do thus once since the anchor boxes depend only
        # on the model configuration, not on the input data
        # For each predictor layer, the tensors for that layer's anchor boxes will have the shape (feature_map_height, feature_map_width, n_boxes, 4)
        self.boxes_list = []    # this will store the anchor boxes for each predictor layer
        self.steps_pre = []     # Horizontal and vertical distance between any two boxes for each predictor layer
        self.wh_list_pre = []   # Box widths and height for each predictor layer
        self.offsets_pre = []   # Offsets for each predictor layer
        self.centers_pre = []   # Anchor box center points as (cy, cx) for each predictor layer

        # Iterate over all predictor layers and compute the anchor boxes for each one
        for i in range(len(self.predictor_sizes)):
            boxes, center, wh, step, offset = self

            self.boxes_list.append(boxes)
            self.wh_list_pre.append(wh)
            self.steps_pre.append(step)
            self.offsets_pre.append(offset)
            self.centers_pre.append(center)

    def __call__(self, ground_truth_labels, diagnostics=False):
        """
        Convert ground truth bounding box data into a suitable format to train an SSD model
        :param ground_truth_labels: (list) A python list of length 'batch_size' that contains one 2D Numpy array for each
                                    batch image. Each such arrays has 'k' rows for the 'k' ground truth bounding boxes belonging
                                    to the respective image, and the data for each ground truth bounding boxes has the format
                                    (class_id, xmin, ymin, xmax, ymax) and 'class_id' must be an integer greater than 0 for all boxes
                                    as class ID is reserved for the background class.
        :param diagnostics:(bool, optional) If True, return the encoded ground truth tensor and a copy of it wit anchor box
                            coordinates in place of the ground truth coordinates. This can be very useful if you want to visualize
                            which anchor boxes got matched to which ground truth boxes
        :return: 'y_encoded', a 3D numpy array of shape (batch_size, boxes, classes + 4 + 4 + 4) that serves as the ground truth
                label tensor for training, where 'boxes' is the total number of boxes predicted by the model prt image, and the classes
                are one-hot-encoded. The four elements after the class vectors in the last axis are the box coordinates, the next
                four elements after that are just dummy elements and the last four elements are the variances
        """

        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4
        batch_size = len(ground_truth_labels)

        # Generate the template for y_encoded
        y_encoded = self

    def generate_anchor_boxes_for_layer(self, feature_map_size, aspect_ratios, this_scale, next_scale, this_steps=None,
                                        this_offsets=None, diagnostics=False):
        """
        Computes an array of the spatial positions and sizes of the anchor boxes for one predictor layer of size 'feature_map_size
        == [feature_map_height, feature_map_width]
        :param feature_map_size:(tuple)  A list or tuple [feature_map_height, feature_map_width] with the spatial dimension
                of the feature map for which to generate the anchor boxes
        :param aspect_ratios:(list) A list of floats, the aspect ratios for which anchor oxes are to be generated. All list elements must be unique
        :param this_scale: (float) A float in [0, 1], the scaling factor for the size of the generate anchor boxes as a fraction of the shorter
                side of the input image
        :param next_scale: (float) A float in [0, 1], the next larger scaling factor. Only relevant if self.two_boxes_for_ar1==True
        :param this_steps:
        :param this_offsets:
        :param diagnostics: (bool, optional) If True, the following additional outputs will be returned:
                1) A list of the center point 'x' and 'y' coordinates for each spatial location.
                2) A list containing (width, height) for each box aspect ratio
                3) A tuple containing (step_height, step_width)
                4) A tuple containing (offset_height, offset_width)
        :return: A 4D numpy tensor of shape (feature_map_height, feature_map_width, n_boxes_per_cell, 4) where the last dimension
        contains (xmin, xmax, ymin, ymax) for each anchor box in each cell of the feature map
        """
        # compute box width and height for each aspect ratio

        # The shorter side of the image will be used to compute 'w' and 'h' using 'scale' and 'aspect_ratios'
        size = min(self.img_height, self.img_width)

        # compute the box width and height for all aspect ratios
        wh_list = []
        for ar in aspect_ratios:
            if ar == 1:
                # compute the regular anchor box for aspect ratio 1
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # compute one slightly larger version using the geometric mean of this scale value and the next
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratio
        # compute the step sizes




