from SSD.bounding_box_utils import iou, convert_coordinates
import numpy as np
from SSD.config import img_height, img_width, classes, aspect_ratios_per_layer, li_steps, offsets, var
from SSD.ssd_encoder.matching_utils import match_multi, match_bipartite


class SSDInputEncoder:
    """
    Transforms ground truth labels for object detection in images(2D bounding box coordinates and class labels) to the
    format required for training an SSD model

    In the process of encoding the ground truth labels, a template of anchor boxes is being built, which are subsequently matched
    to the ground truth boxes via an intersection-over-union threshold criterion
    """
    def __init__(self, predictor_sizes, min_scale=0.1, max_scale=0.9, scales=None, clip_boxes=False, two_boxes_for_ar1=True,
                 pos_iou_threshold=0.5, neg_iou_limit=0.3, border_pixels='half', normalize_coords=True,
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
        :param pos_iou_threshold: (float, optional) The intersection-over-union similarity threshold that must be met in order to match
                a given ground truth box to a given anchor box
        :param neg_iou_limit: (float, optional) The maximum allowed intersection-over-union similarity of an anchor box with
                any ground truth box to be labeled a negative box (background). If an anchor box is neither a positive, nor a negative box
                it will be ignored during training
        :param border_pixels:(str, optional) How to treat the border pixels of the bounding boxes. Can be 'include', 'exclude',
                or 'half'. If 'include', the border pixels of the bounding boxes belong to the boxes. If 'exclude', the border pixels
                do not belong to the boxes. If 'half', then one of each of two horizontal and vertical borders belong to the boxes,
                but not the other.
        :param normalize_coords: (bool, optional) If 'True', the encoder uses relative instead of absolute coordinates. This means
                instead of using absolute target coordinates, the encoder will scale all coordinates to be within [0, 1]. Those
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
        self.steps = li_steps
        self.offsets = offsets
        self.clip_boxes = clip_boxes
        self.variances = var
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
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
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                                                   aspect_ratios=self.aspect_ratios[i],
                                                                                   this_scale=self.scales[i],
                                                                                   next_scale=self.scales[i+1],
                                                                                   this_steps=self.steps[i],
                                                                                   this_offsets=self.offsets[i])

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
        y_encoded = self.generate_encoding_template(batch_size=batch_size)

        # Match ground truth boxes to anchor boxes. Every anchor box that does not have a ground truth match and the maximal IoU
        # overlap with any ground truth box is less than or equal to 'neg_iou_limit' will be a negative box
        y_encoded[:, :, self.background_id] = 1     # all boxes are background boxes by default
        class_vectors = np.eye(self.n_classes)      # an identity matrix that we'll use as one-hot class vectors

        for i in range(batch_size):
            # if there is no ground truth for this batch item, there is nothing to match
            if ground_truth_labels[i].size == 0:
                continue
            labels = ground_truth_labels[i].astype[np.float]

            # check for degenerate ground truth bounding boxes before attempting any computations
            if np.any(labels[:, [xmax]] - labels[:, [xmin]] <= 0) or np.any(labels[:, [ymax]] - labels[:, [ymin]] <= 0):
                raise ValueError("SSDInputEncoder detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, labels))

            # Normalize the box coordinates
            if self.normalize_coords:
                labels[:, [ymin, ymax]] /= self.img_height
                labels[:, [xmin, xmax]] /= self.img_width

            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)]     # The one-hot class IDs for the ground truth boxes of this batch item
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]], axis=-1)

            # compute the IoU similarities between all anchor boxes and all ground truth boxes for this batch item
            # This is a matrix of shape (num_ground_truth_boxes, num_anchor_boxes)
            similarities = iou(labels[:, [xmin, ymin, xmax, ymax]], y_encoded[i, :, -12:-8], coords='corners', mode='outer_product', border_pixels=self.border_pixels)

            # Match each ground truth box to the one anchor box with the highest IoU. This ensures that each ground truth box
            # will have at least one good match

            # For each ground truth box, get the anchor box to match with it
            bipartite_matches_indices = match_bipartite(inputs=similarities)

            # apply ground truth data to the matched anchor boxes
            y_encoded[i, bipartite_matches_indices, :-8] = labels_one_hot

            # set the columns of the matched anchor boxes to zero to indicate that they were matched
            similarities[:, bipartite_matches_indices] = 0

            # Second: Each remaining anchor box will be matched to its most similar ground truth box with an IoU of at least 'pos_iou_threshold'
            # or not matched if there is no ground truth box

            # Get all matches that satisfy the IoU threshold
            matches = match_multi(inputs=similarities, threshold=self.pos_iou_threshold)

            # apply the ground truth data to the matched anchor boxes
            y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]

            # set the columns of the matched anchor boxes to zero to indicate that they were matched
            similarities[:, matches[1]] = 0

            # Third: Now after the matching is done, all negative (background) anchor boxes that have an IoU of 'neg_iou_limit'
            # or more with any ground truth box will be set to neutral

            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        y_encoded[:, :, -12:-8] -= y_encoded[:, :, -8:-4]   # (gt - anchor) for all four coordinates
        y_encoded[:, :, [-12, -10]] /= np.expand_dims(y_encoded[:, :, -6] - y_encoded[:, :, -8], axis=-1)   # (xmin(gt) - xmin(anchor)) / w(anchor)
        y_encoded[:, :, [-11, -9]] /= np.expand_dims(y_encoded[:, :, -5] - y_encoded[:, :, -7], axis=-1)    # (ymin(gt) - ymin(anchor)) / h(anchor)
        y_encoded[:, :, -12:-8] = y_encoded[:, :, -4:]  # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w anh h respectively

        return y_encoded

    def generate_anchor_boxes_for_layer(self, feature_map_size, aspect_ratios, this_scale, next_scale, this_steps=None,
                                        this_offsets=None):
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
        if this_steps is None:
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps

        # compute the offsets
        if this_offsets is None:
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets

        # Compute  the grid of anchor box center points
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)   # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create 4D tensor template of shape (feature_map_height, feature_map_width, n_boxes, 4)
        # where the last dimension will contain (cx, cy, w, h)
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))    # set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))    # set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]    # set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]    # set h

        # convert (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If clip_boxes is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords
            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        # normalize_coords is enabled, normalize the coordinates to be within [0, 1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        return boxes_tensor

    def generate_encoding_template(self, batch_size):
        """
        Produces an encoding template for the ground truth label tensor for a given batch
        :param batch_size: (int) the batch size
        :return: a numpy array of shape (batch_size, n_boxes, classes + 12), the template into which to encode the ground truth labels
                for training. The last axis has length "classes + 12" because the model output contains 4 predicted box coordinate offsets,
                the 4 coordinates for the anchor boxes and the 4 variance values
        """

        # Tile the anchor boxes for each predictor layer across all batch items
        boxes_batch = []
        for boxes in self.boxes_list:
            # add one dimension to self.boxes_list to account for the batch_size and tile it along
            # The result will be a 5D tensor of shape (batch_size, feature_map_height, feature_map_width, n_boxes, 4)
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # reshape the 5D tensor above into a 3D tensor of shape (batch_size, feature_map_height * feature_map_width * n_boxes, 4)
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        # concatenate the anchor tensors from the individual layers to one
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        # 3: Create a template tensor to hold the one-hot class encoding for shape (batch_size, n_boxes, classes)
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: create a tensor to contain variances. This tensor has the same shape as 'boxes_tensor' and simply contain the
        # same variances value for every position in the last axis
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances

        # concatenate the classes, boxes and variances tensors to get our final template for y_encoded
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        return y_encoding_template
