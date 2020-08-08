import tensorflow as tf


def smooth_loss(y_true, y_pred):
    """
        Compute smooth L1 loss
        :param y_true: Tensor which have shape of (batch_size, num_boxes, 4) containing the ground truth bounding box coordinates.
                        Here, 4 is (xmin, xmax, ymin, ymax)
        :param y_pred: Contain the predicted bounding box coordinates
        :return: The smooth L1 loss have shape of (batch, num_boxes_total)
        """
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2

    loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(loss, axis=-1)


def log_loss(y_true, y_pred):
    """
    Compute the softmax log loss
    :param y_true: a tensor of shape (batch_size, num_boxes, num_classes) contains the ground truth bounding box
                    categories
    :param y_pred: the predicted bounding box categories (batch_size, num_boxes, num_classes)
    :return: The softmax log loss of shape (batch, num_boxes_total)
    """
    # make sure that y_pred doesn't contain any zeros and too big
    y_pred = tf.math.maximum(tf.math.minimum(y_pred, 1 - 1e-10), 1e-10)
    softmax_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    return softmax_loss


def multi_box_loss(y_true, y_pred, alpha=1.0, neg_pos_ratio=3, n_neg_min=0):
    """

    :param n_neg_min: the minimum ratio of negative ground truth boxes. This argument can be used to make sure that model
                        learns from a minimum number of negatives in batches
    :param y_true: ground truth targets of shape (batch_size, num_boxes, num_classes + 4 + 8). The last axis contain
                    [classes one-hot encoded, 4 ground truth box coordinate offset, 8 arbitrary entries]
    :param y_pred: a numpy array of shape (batch_size, num_boxes, num_classes + 4 + 8)
    :param alpha: a factor to weight the localization loss in the computation
    :param neg_pos_ratio: the maximum ratio of negative to positive truth boxes to include in the loss computation
    :return: a scalar, the total multitask loss for classification and localization
    """
    neg_pos_ratio = tf.constant(neg_pos_ratio)
    n_neg_min = tf.constant(n_neg_min)
    alpha = tf.constant(alpha)

    batch_size = tf.shape(y_pred)[0]
    n_boxes = tf.shape(y_pred)[1]

    # 1: Compute the losses for class and box predictions for every box
    classification_loss = log_loss(y_true[:, :, -12], y_pred[:, :, -12])            # output shape: (batch_size, n_boxes)
    localization_loss = smooth_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8])     # output shape: (batch_size, n_boxes)

    # 2: Compute the classification loss fot all negative default boxes

    # create masks for the positive and negative ground truth classes
    negatives = y_true[:, :, 0]     # have shape of (batch_size, num_boxes)
    positives = tf.reduce_max(y_true[:, :, 1:-12], axis=-1)    # have shape of (batch_size, num_boxes)
    positives = tf.cast(positives, dtype=tf.float32)

    # count the number of positive boxes (classes 1 to n) in y_true across the whole batch
    n_positive = tf.reduce_sum(positives)

    # sum up losses for the positive boxes per batch item
    pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)    # have shape of (batch_size, )

    # compute the classification loss for all negative boxes
    neg_class_loss_all = classification_loss * negatives    # have shape of (batch_size, n_boxes)
    n_neg_losses = tf.math.count_nonzero(neg_class_loss_all, dtype=tf.int32)

    # compute the number of negative examples we want to account for in the loss
    # we will keep at most 'neg_pos_ratio' times the number of positives in y_true, but at least n_neg_min
    n_negative_keep = tf.minimum(tf.maximum(neg_pos_ratio * tf.cast(n_positive, dtype=tf.int32), n_neg_min), n_neg_losses)

    # In the case that no negative ground truth boxes at all or the classification loss for all negative boxes is zero,
    # return zero as the 'neg_class_loss'

    # Case 1: No negative ground truth boxes at all
    def case1():
        return tf.zeros([batch_size])

    def case2():
        # pick the top-k(k == n_negative_keep) boxes with the highest confidence loss that belong to the background class
        # in the ground truth data

        # reshape neg_class_loss_all from (batch_size, num_boxes) to (batch_size * num_boxes,)
        neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])

        # get the indices of the n_negative_keep boxes with the highest loss out of those
        values, indices = tf.nn.top_k(neg_class_loss_all, k=n_negative_keep, sorted=False)

        # create a mask with these indices, output shape of (batch_size * n_boxes, )
        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32),
                                       shape=tf.shape(neg_class_loss_all_1D))
        negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, n_boxes]), dtype=tf.float32)

        # apply mask to all other classification losses
        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)   # tensor of shape (batch_size, )

        return neg_class_loss

    negative_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), case1, case2)
    class_loss = pos_class_loss + negative_class_loss

    # 3. Compute the localization loss for the positive target
    loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)    # tensor of shape (batch_size, )
    total_loss = (class_loss + alpha * loc_loss) / tf.maximum(1.0, n_positive)
    total_loss = total_loss * tf.cast(batch_size, dtype='float32')

    return total_loss


