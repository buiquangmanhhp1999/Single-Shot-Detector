import numpy as np
import inspect
import sklearn.utils as utils
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import pickle
import cv2
import sys
from tqdm import tqdm, trange
import h5py
from config import class_names
from BoundGenerator import BoxFilter
from data_utils import create_train_img_file, create_val_img_file
import os


class DataGenerator:
    """
    A generator to generate batches of samples and corresponding labels indefinitely
    - Can shuffle the dataset consistently after each complete pass
    - Can perform image transformations for data conversion and data augmentation
    """

    def __init__(self, trainable=True, load_images_into_memory=False, filenames=None, verbose=True):
        """
        Initialize the data generator. You can either load a dataset directly here in the constructor,
        e.g an HDF5 dataset or you can use xml parser method to read in a dataset

        :param load_images_into_memory: (bool, optional) If 'True', the entire dataset will be loaded into memory
        :param filenames:(string): contains the filename (full paths) of the images to be used. Each line of the text contains
                                    filename to one image
        :param verbose: If True, prints out the progress for some constructor operations that may take a bit longer
        """
        self.labels_format = {'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}
        self.load_images_into_memory = load_images_into_memory

        if filenames is None:
            if trainable:
                filenames = "/data/train_img_path.txt"
                create_train_img_file(name=filenames)
            else:
                filenames = "/data/val_img_path.txt"
                create_val_img_file(name=filenames)

        # read data from filename
        if filenames is None:
            ValueError("filenames must be string. Not be None")

        with open(filenames, "rb") as f:
            self.filenames = [line.strip() for line in f]

        self.datasets_size = len(self.filenames)
        self.dataset_indices = np.arange(self.datasets_size, dtype=np.int32)

        if self.load_images_into_memory:
            self.images = []
            if verbose:
                iteration = tqdm(self.filenames, desc='Loading image into memory', file=sys.stdout)
            else:
                iteration = self.filenames

            for filename in iteration:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.images = None

        self.labels = None
        self.hdf5_dataset = None
        self.hdf5_dataset_path = None

    def read_data_from_xml(self, datasets_dir=None, verbose=True):
        """
        This is an XML parser for the Pascal VOC datasets
        :param verbose: If True, prints out the progress for operations that may take a bit longer
        :param datasets_dir: this directory contains all annotation file and image file
        :return: images, image filenames, labels
        """
        path = Path(datasets_dir)
        self.filenames = []
        self.labels = []

        # loop over all file xml in this dataset
        for file_xml in path.glob("*.xml"):
            root = ET.parse(str(file_xml)).getroot()
            file_name = root.find('filename').text
            folder = datasets_dir + file_name
            self.filenames.append(folder)
            # We'll store all boxes for this image here
            boxes = []

            # parse the data for each object
            for obj in root.iter('object'):
                obj_name = obj.find('name').text
                if obj_name == 'image':
                    continue
                class_id = class_names.index(obj_name)

                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                boxes.append([class_id, xmin, ymin, xmax, ymax])
            self.labels.append(boxes)

        self.datasets_size = len(self.filenames)
        self.dataset_indices = np.arange(self.datasets_size)

        if self.load_images_into_memory:
            self.images = []
            if verbose:
                iterate = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                iterate = self.filenames
            for file_name in iterate:
                with Image.open(file_name) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

    def create_hdf5_dataset(self, resize=None, file_path=None, variable_img_size=True, verbose=True):
        """
        Convert the currently loaded dataset into a HDF5 file. This HDF5 file contains all images as uncompressed arrays in a
        contiguous block of memory, which allows for them to be loaded faster. But may take up considerably more space on
        your hard drive than the sum of the source images in a compressed format such as JPG or PNG
        :param file_path: (str, optional) The full file path to store HDF5 dataset. You can load this output file via the
                          "DataGenerator" constructor in the future
        :param resize:(tuple, optional) 'False' or tuple (height, width) that represents the target size for the images
        :param variable_img_size:(bool, optional) img_size value will be stored in the HDF5 dataset in order ti be able to quickly
                                find out whether the image in the dataset all have the same size or not
        :param verbose:(bool, optional) Whether or not print out the progress of the dataset creation
        :return:
        """
        self.hdf5_dataset_path = file_path
        dataset_size = len(self.filenames)

        # create the hdf5 file
        hdf5_dataset = h5py.File(file_path, 'w')

        # create a few attributes which this dataset contain
        hdf5_dataset.attrs.create(name='has_labels', data=False, shape=None, dtype=np.bool_)

        # don't resize
        if variable_img_size and not resize:
            hdf5_dataset.attrs.create(name='variable_image_size', data=True, shape=None, dtype=np.bool_)
        else:  # resize
            hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

        # create the dataset in which the image will be stored as flattened arrays
        hdf5_images = hdf5_dataset.create_dataset(name='images', shape=(dataset_size,), maxshape=None,
                                                  dtype=h5py.special_dtype(vlen=np.uint8))

        # create the dataset that will hold the image height, widths and channels that we need in order to reconstruct the
        # image from the flattened array later
        hdf5_image_shapes = hdf5_dataset.create_dataset(name='images_shapes', shape=(dataset_size, 3),
                                                        maxshape=(None, 3), dtype=np.int32)

        if self.labels is not None:
            # create the dataset in which the labels will be stored as flattened arrays
            hdf5_labels = hdf5_dataset.create_dataset(name='labels', shape=(dataset_size,), maxshape=None,
                                                      dtype=h5py.special_dtype(vlen=np.int32))
            hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes', shape=(dataset_size, 2),
                                                            maxshape=(None, 2), dtype=np.int32)
            hdf5_dataset.attrs.modify(name='has_labels', value=True)

        if verbose:
            tr = trange(dataset_size, desc='Creating HDF5 dataset', file=sys.stdout)
        else:
            tr = range(dataset_size)

        # Iterate over all images in the dataset
        for i in tr:
            # Store the image
            with Image.open(self.filenames[i]) as image:
                image = np.asarray(image, dtype=np.uint8)

                # make sure all images end up having three channels
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3:
                    if image.shape[2] == 1:
                        image = np.concatenate([image] * 3, axis=-1)
                    elif image.shape[2] == 4:
                        image = image[:, :, :3]

                image = cv2.resize(image, (300, 300))
                if resize:
                    image = cv2.resize(image, dsize=(resize[1], resize[0]))

                # flatten the image array and write it to the image dataset
                hdf5_images[i] = image.reshape(-1)

                # write the image's shape to the image shapes dataset.
                hdf5_image_shapes[i] = image.shape

            # store the ground truth if we have any
            if self.labels is not None:
                labels = np.asarray(self.labels[i])
                # flatten the labels array and write it to the labels datasets
                hdf5_labels[i] = labels.reshape(-1)
                # write the label's shape to the label shapes dataset
                hdf5_label_shapes[i] = labels.shape

        hdf5_dataset.close()
        self.hdf5_dataset = h5py.File(file_path, "r")
        self.hdf5_dataset_path = file_path
        self.datasets_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.datasets_size)

    def load_hdf5_dataset(self, file_path=None, verbose=True):
        """
        Loads an HDF5 dataset that is in the format that "create_hdf5_dataset()' method produces
        :param file_path: path to file h5 dataset
        :param verbose: (bool, optional) If True, prints out the progress while loading the dataset
        :return: None
        """
        self.hdf5_dataset = h5py.File(file_path, "r")
        self.datasets_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.datasets_size)

        if self.load_images_into_memory:
            self.images = []
            if verbose:
                tr = trange(self.datasets_size, desc='Loading images to memory', file=sys.stdout)
            else:
                tr = trange(self.datasets_size)
            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['images_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']

            if verbose:
                tr = trange(self.datasets_size, desc="Loading labels", file=sys.stdout)
            else:
                tr = range(self.datasets_size)

            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

    def generate(self, transformations, batch_size=32, shuffle=True, label_encoder=None, returns=None,
                 keep_images_without_gt=False):
        """
        Generates batches of samples and corresponding labels indefinitely
        Can shuffle the samples consistently after each complete pass
        :param transformations: (list, optional) A list of transformations that will be applied to the images abd labels
                                in the given order. Each transformation is a callable that takes as input an image and optionally
                                labels and returns an image and optionally labels in the same format
        :param batch_size: (int, optional) The size of the batches to be generated
        :param shuffle: Whether or not to shuffle the dataset before each pass
        :param label_encoder: (callable, optional): Only relevant if labels are given. A callable that takes as input the labels
                                of batch and returns some structure that represents those labels. The general use case for this to convert labels
                                from their input format to a format that a given object detection model needs as its training targets
        :param returns: A set of strings that determines what outputs the generator yields. The output tuple can contain the following outputs
                        according to the specified keyword strings
                         * 'processed_images': An array containing the processed images.
                         * 'encoded_labels': The encoded labels tensor.
                         * 'filenames': A list containing the file names(full paths) of the images in the batch
                         * 'inverse_transform': A nested list that contains a list of 'inverter' functions for each item in the
                            batch. These inverter function take (predicted) labels for an image as input and apply the inverse of the
                            inverse of the transformation that were applied to the original image to them.
                         * 'original_images': A list containing the original images in the batch before any processing
                         * 'original_labels': A list containing the original ground truth boxes for the images in this batch before
                            any processing.
        :param keep_images_without_gt:(bool, optional) If False, images for which there aren't any ground truth boxes before any transformations
                                        have been applied will be removed from the batch. If 'True', such images will beb kept in
                                        the batch
        :return: The next batch as a tuple of items as defined by the 'returns' arguments
        """
        if returns is None:
            returns = {'processed_images', 'encoded_labels'}

        if shuffle:
            object_to_shuffle = [self.dataset_indices]
            if self.filenames is not None:
                object_to_shuffle.append(self.filenames)
            if self.labels is not None:
                object_to_shuffle.append(self.labels)
            shuffled_object = utils.shuffle(*object_to_shuffle)
            for i in range(len(object_to_shuffle)):
                object_to_shuffle[i][:] = shuffled_object[i]

        # Override the labels formats of all the transformation to make
        if self.labels is not None:
            for transform in transformations:
                transform.labels_format = self.labels_format

        # Generate mini batches
        current = 0
        while True:
            batch_X, batch_y = [], []

            if current >= self.datasets_size:
                current = 0

                if shuffle:
                    object_to_shuffle = [self.dataset_indices]
                    if self.filenames is not None:
                        object_to_shuffle.append(self.filenames)
                    if self.labels is not None:
                        object_to_shuffle.append(self.labels)
                    shuffled_object = utils.shuffle(*object_to_shuffle)
                    for i in range(len(object_to_shuffle)):
                        object_to_shuffle[i][:] = shuffled_object[i]

            # ==================================================================
            # Get the image. labels for this batch
            # ==================================================================
            # we prioritize our option in the following order:
            # 1) If we have the images, already loaded in memory, get them from there
            # 2) Else, if we have an HDF5 dataset, get the images from there
            # 3) Else, if we have neither of the above, we'll have to load the individual image files from disk

            batch_indices = self.dataset_indices[current:current + batch_size]
            if self.images is not None:
                for i in batch_indices:
                    batch_X.append(self.images[i])
                if self.filenames is not None:
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            elif self.hdf5_dataset is not None:
                for i in batch_indices:
                    batch_X.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['images_shapes'][i]))
                if self.filenames is not None:
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.filenames[current: current + batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            if self.labels is not None:
                batch_y = deepcopy(self.labels[current:current + batch_size])
            else:
                batch_y = None

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X)
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y)

            current += batch_size

            # Perform image transformation
            batch_items_to_remove = []  # In case the transform we need to remove any images from the batch, store their indices in this list
            batch_inverse_transforms = []

            for i in range(len(batch_X)):
                if batch_y is not None:
                    # convert the labels for this image to an array
                    batch_y[i] = np.array(batch_y[i])

                    # if this image has no ground truth boxes, maybe we don;t want to keep it in the batch
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                # apply any image transformations we may have received
                if transformations:
                    inverse_transforms = []

                    for transform in transformations:
                        if self.labels is not None:
                            if ('inverse_transform' in returns) and (
                                    'return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i],
                                                                                      return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[i] is None:
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue
                        else:
                            if ('inverse_transform' in returns) and (
                                    'return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                # Check for degenerate boxes in this batch item
                if self.labels is not None:
                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    if np.any(batch_y[i][:, xmax] - batch_y[i][:, xmin] <= 0) or np.any(
                            batch_y[i][:, ymax] - batch_y[i][:, ymin] <= 0):
                        # remove degenerate box
                        batch_y[i] = BoxFilter(check_overlap=False, check_min_area=False, check_degenerate=True,
                                               labels_format=self.labels_format)(batch_y[i])
                        if (batch_y[i].size == 0) and not keep_images_without_gt:
                            batch_items_to_remove.append(i)

            # remove any items we might not want to keep from the batch
            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms:
                        batch_inverse_transforms.pop(j)
                    if self.labels is not None:
                        batch_y.pop(j)
                    if 'original_images' in returns:
                        batch_original_images.pop(j)
                    if 'original_labels' in returns and (self.labels is not None):
                        batch_original_labels.pop(j)

            batch_X = np.array(batch_X)

            # if we have a label encoder, encode our label
            if label_encoder is not None or self.labels is None:
                batch_y_encoded = label_encoder(batch_y, diagnostics=True)
            else:
                batch_y_encoded = None

            # Compose the output
            ret = []

            if 'processed_images' in returns:
                ret.append(np.array(batch_X))
            if 'encoded_labels' in returns:
                ret.append(np.array(batch_y_encoded))
            if 'processed_labels' in returns:
                ret.append(batch_y)
            if 'filenames' in returns:
                ret.append(batch_filenames)
            if 'original_images' in returns:
                ret.append(batch_original_images)
            if 'original_labels' in returns:
                ret.append(batch_original_labels)

            yield ret

    def save_dataset(self, filenames_path='filenames.pkl', labels_path=None):
        """
        Writes the current `filenames`, `labels` to the specified files.
        This is particularly useful for large datasets with annotations that are
        parsed from XML files, which can take quite long. If you'll be using the
        same dataset repeatedly, you don't want to have to parse the XML label
        files every time.
        Arguments:
            filenames_path (str): The path under which to save the filenames pickle.
            labels_path (str): The path under which to save the labels pickle.
        """
        with open(filenames_path, 'wb') as f:
            pickle.dump(self.filenames, f)
        if labels_path is not None:
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels, f)

    def get_dataset(self):
        """
        Returns:
            4-tuple containing lists and/or `None` for the filenames, labels, image IDs,
            and evaluation-neutrality annotations.
        """
        return self.filenames, self.labels

    def get_dataset_size(self):
        """
        Returns:
            The number of images in the dataset.
        """
        return self.datasets_size
