import numpy as np


def match_bipartite(inputs):
    """
    The algorithm works as follows:
    The first axis of inputs represent ground truth boxes and the second axis of inputs represent anchor boxes

    the ground truth box that has the greatest similarity with any anchor box will be matched first, then out of the remaining
    ground truth boxes, the ground truth box that has the greatest similarity with any of the remaining anchor boxes will matched
    second, and so on. The ground truth boxes will be matched in descending order by maximum similarity with any of the respectively
    remaining anchor boxes
    :param inputs: a 2D numpy array (m, n). And m <= n
    :return: an 1D numpy array of length m that represents the matched index along the second axis of inputs for each index
    along the first axis
    """
    inputs = np.copy(inputs)
    num_ground_truth_boxes = inputs.shape[0]
    all_ground_truth_indices = list(range(num_ground_truth_boxes))

    # the 1D array contain the matched anchor box
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # in each iteration of the loop below, exactly one ground truth box will be matched to one anchor box
    for _ in range(num_ground_truth_boxes):
        # find the maximal anchor-ground truth pair in two steps:
        # 1. reduce over the anchor boxes
        # 2. reduce over the ground truth boxes
        anchor_indices = np.argmax(inputs, axis=1)      # Reduce along the anchor box axis.
        overlap = inputs[all_ground_truth_indices, anchor_indices]
        ground_truth_index = np.argmax(overlap)         # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index

        # set the row of the matched ground truth box and the column of the matched anchor box to all zero. This ensures that those
        # boxes will not be matched again, because they will never be the best matches for any other boxes
        inputs[ground_truth_index] = 0
        inputs[:, anchor_index] = 0

    return matches


def match_multi(inputs, threshold):
    """
    Match all elements along the second axis of 'inputs' to their best matches along the first axis subject to the constraint that
    the inputs of a match must be greater than or equal to 'threshold' in order to produce a match
    :param inputs:
    :param threshold:
    :return: Two 1D numpy arrays of equal length that represent the matched indices. The first array contains the indices along the first axis
    of 'inputs', the second array contains the indices along the second axis
    """
    num_anchor_boxes = inputs.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes))

    # find the best ground truth match for every anchor box
    ground_truth_indices = np.argmax(inputs, axis=0)    # array of shape (inputs.shape[1],)
    overlaps = inputs[ground_truth_indices, all_anchor_indices]

    # filter out the matches with inputs below a threshold
    anchor_indices_thresh_met = np.nonzero(overlaps > threshold)[0]
    ground_truth_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return ground_truth_indices_thresh_met, anchor_indices_thresh_met
