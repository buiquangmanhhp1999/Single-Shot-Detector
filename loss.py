import tensorflow as tf


class SSDLoss:
    def __init__(self, neg_pos_ratio=3, n_neg_min=0, alpha=1.0):
        """
        :param neg_pos_ratio: The maximum ratio of negative to positive ground truth boxes
        :param n_neg_min: The minimum number of negative ground truth boxes to enter the loss computation per batch
        :param alpha: A factor to weight the localization loss in the computation of the total loss
        """
