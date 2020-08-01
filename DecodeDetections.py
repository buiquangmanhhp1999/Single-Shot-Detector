import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Layer


class DecodeDetections(Layer):
    """
    A Keras layer to decode the raw SSD prediction output

    Input shape:
        3D tensor of shape (batch_size, n_boxes, n_classes + 12)

    Output shape:
        3D tensor of shape (batch_size, top_k, 6)
    """

    def __init__(self, confidence_thresh=0.01, iou_thresh=0.45, top_k=200, nms_max_output_size=400, coords='centroids',
                 normalize_coords=True, img_height=None, img_width=None, **kwargs):
        """
        :param confidence_thresh: A float in [0, 1), the minimum classification confidence in a specific positive class
                in order to be considered for the non-maximum suppression stage for the respective class
        :param iou_thresh: A float in [0, 1). All boxes have a value of greater than iou_thresh will be removed form the set
                of prediction for a given class
        :param top_k: (int, opt) The number of highest scoring prediction to be kept for each batch item after the non-maximum
                        suppression stage
        :param nms_max_output_size: (int, opt) The maximum number of predictions that will be left after performing non-maximum
                suppression
        :param coords: Must be 'centroids'
        :param normalize_coords: (bool, opt) Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
                and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
                relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
                Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
                coordinates. Requires `img_height` and `img_width` if set to `True`.
        :param img_height: The height of the input images. Only needed if `normalize_coords` is `True`.
        :param img_width: The width of the input images. Only needed if `normalize_coords` is `True`.
        :param kwargs:
        """

        if normalize_coords and ((img_height is None) or (img_width is None)):
            raise ValueError(
                "If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(
                    img_height, img_width))

        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.top_k = top_k
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width
        self.coords = coords
        self.nms_max_output_size = nms_max_output_size

        # We need these members for TensorFlow.
        self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
        self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
        self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')
        super(DecodeDetections, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetections, self).build(input_shape=input_shape)

    def call(self, y_pred, mask=None):
        """
        :param y_pred: 3D tensor of shape (batch, n_boxes_total, n_classes + 4 + 8)
        :param mask:
        :return: 3D tensor of shape (batch_size, top_k, 6). The second axis is zero-padded to always yield 'top_k' predictions
                per batch items. The last axis contains the coordinates for each predicted box in the format
                [class_id, confidence, xmin, ymin, xmax, ymax]
        """

        # 1. convert the box coordinate from predicted anchor box offsets to predicted absolute coordinates
        cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[
            ..., -8]  # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cy = y_pred[..., -11] * y_pred[..., -3] * y_pred[..., -5] + y_pred[
            ..., -7]  # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        w = tf.exp(y_pred[..., -10] * y_pred[..., -2]) * y_pred[..., -6]  # w = exp(w_pred * variance_w) * w_anchor
        h = tf.exp(y_pred[..., -9] * y_pred[..., -1]) * y_pred[..., -5]  # h = exp(h_pred * variance_h) * h_anchor

        # Convert 'centroids' to 'corners'.
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        if self.tf_normalize_coords:
            xmin = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
            ymin = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
            xmax = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
            ymax = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
        else:
            xmin = tf.expand_dims(xmin, axis=-1)
            ymin = tf.expand_dims(ymin, axis=-1)
            xmax = tf.expand_dims(xmax, axis=-1)
            ymax = tf.expand_dims(ymax, axis=-1)

        # Concatenate the one-hot class confidences and the converted box coordinates to form the decoded predictions tensor.
        y_pred = tf.concat(values=[y_pred[..., :-12], xmin, ymin, xmax, ymax], axis=-1)

        # 2. Perform confidence thresholding, per-class non-maximum suppression and top-k filtering
        n_classes = y_pred.shape[2] - 4

        # Create a function that filters the predictions for the given batch item. Specifically, it performs:
        # - confidence thresholding
        # - non-maximum suppression (NMS)
        # - top-k filtering

        def filter_prediction(batch_item):
            # create a function that filters the predictions for one single class
            def filter_single_class(index):
                # from a tensor of shape (n_boxes, n_classes + 4 coordinates) extract a tensor of shape (n_boxes, 1 + 4 coordinates)
                # that contains the confidence value for just one class, determined by 'index'

                confidences = tf.expand_dims(batch_item[..., index], axis=-1)
                class_id = tf.fill(dims=tf.shape(confidences), value=tf.cast(index, name='float'))
                box_coordinates = batch_item[..., -4:]

                single_class = tf.concat([class_id, confidences, box_coordinates], axis=-1)

                # Apply confidence threshold
                threshold_mask = single_class[:, 1] > self.tf_confidence_thresh
                single_class = tf.boolean_mask(tensor=single_class, mask=threshold_mask)

                # If any boxes made the threshold, perform NMS
                def perform_nms():
                    scores = single_class[..., 1]

                    # tf.image.non_max_suppression() needs the box coordinates in the format (ymin, xmin), (ymax, xmax)
                    xmin_nms = tf.expand_dims(single_class[..., -4], axis=-1)
                    ymin_nms = tf.expand_dims(single_class[..., -3], axis=-1)
                    xmax_nms = tf.expand_dims(single_class[..., -2], axis=-1)
                    ymax_nms = tf.expand_dims(single_class[..., -1], axis=-1)
                    boxes = tf.concat(values=[ymin_nms, xmin_nms, ymax_nms, xmax_nms])

                    maxima_indices = tf.image.non_max_suppression(boxes=boxes, scores=scores,
                                                                  max_output_size=self.tf_nms_max_output_size,
                                                                  iou_threshold=self.iou_thresh,
                                                                  name='non_maximum_suppression')

                    maxima = tf.gather(params=single_class, indices=maxima_indices, axis=0)

                    return maxima

                def no_confident_prediction():
                    return tf.constant(value=0, shape=(1, 6))

                single_class_nms = tf.cond(tf.equal(tf.size(single_class), 0), no_confident_prediction, perform_nms)

                # Make sure single_class is exactly self.nms_max_output_size elements long
                padded_single_class = tf.pad(tensor=single_class_nms,
                                             paddings=[[0, self.tf_nms_max_output_size - tf.shape(single_class_nms)[0]],
                                                       [0, 0]],
                                             mode='CONSTANT', constant_values=0)

                return padded_single_class

            # Iterate filter_single_class() over all class indices
            filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i), elems=tf.range(1, n_classes),
                                                dtype=tf.float32,
                                                parallel_iterations=128, back_prop=False, swap_memory=False,
                                                infer_shape=True, name='loop_over_classes')

            # concatenate the filtered results for all individual classes to one tensor
            filtered_prediction = tf.reshape(tensor=filtered_single_classes, shape=(-1, 6))

            # perform top-k filtering for this batch item or pad it in case there are fewer than 'self.top_k' boxes left at this
            # point. Either way, produce a tensor of length 'self.top_k'.
            def top_k():
                return tf.gather(params=filtered_prediction,
                                 indices=tf.nn.top_k(filtered_prediction[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=filtered_prediction,
                                            paddings=[[0, self.tf_top_k - tf.shape(filtered_prediction)[0]], [0, 0]],
                                            mode='CONSTANT', constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_prediction)[0], self.tf_top_k), top_k,
                                  pad_and_top_k)

            return top_k_boxes

        # Iterate filter_predictions() over all batch items
        output_tensor = tf.map_fn(fn=lambda x: filter_prediction(x), elems=y_pred, dtype=None, parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False, infer_shape=True, name='loop_over_batch')

        return output_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        return batch_size, self.tf_top_k, 6

    def get_config(self):
        config = {
            'confidence_thresh': self.confidence_thresh,
            'iou_threshold': self.iou_threshold,
            'top_k': self.top_k,
            'nms_max_output_size': self.nms_max_output_size,
            'coords': self.coords,
            'normalize_coords': self.normalize_coords,
            'img_height': self.img_height,
            'img_width': self.img_width,
        }
        base_config = super(DecodeDetections, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
