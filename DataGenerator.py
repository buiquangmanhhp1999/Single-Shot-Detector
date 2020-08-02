import numpy as np
import inspect
from collections import defaultdict
import sklearn.utils as utils
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import sys
from tqdm import tqdm, trange
import h5py
from SSD.config import labels_output_format, class_names
from SSD.BoundGenerator import BoxFilter


class DataGenerator:
    """
    A generator to generate batches of samples and corresponding labels indefinitely
    - Can shuffle the dataset consistently after each complete pass
    - Can perform image transformations for data conversion and data augmentation
    """

    def __init__(self, load_images_into_memory=False, hdf5_dataset_path=None, filenames=None, labels=None, verbose=True):
        """
        Initialize the data generator. You can either load a dataset directly here in the constructor,
        e.g an HDF5 dataset or you can use xml parser method to read in a dataset
        :param load_images_into_memory: (bool, optional) If 'True', the entire dataset will be loaded into memory
        :param hdf5_dataset_path: (str, optional) The full file path of an HDF5 file that contains a dataset in te format
                that contains a datasets in the format that the "create_hdf5_dataset()" method.
        :param filenames:(string): contains the filename (full paths) of the images to be used. Each line of the text contains
                                    filename to one image
        :param verbose: If True, prints out the progress for some constructor operations that may take a bit longer
        """
        self.labels_output_format = labels_output_format
        self.labels_format = {'class_id': labels_output_format.index('class_id'),
                              'xmin': labels_output_format.index('xmin'),
                              'ymin': labels_output_format.index('ymin'),
                              'xmax': labels_output_format.index('xmax'),
                              'ymax': labels_output_format.index('ymax')}
        self.load_images_into_memory = load_images_into_memory

        if filenames is None:
            raise ValueError("The filename is empty")

        # read data from filename
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

        if isinstance(labels, dict):
            self.labels = labels
        else:
            raise ValueError('labels must be python dictionary')

        if hdf5_dataset_path is not None:
            self.hdf5_dataset_path = hdf5_dataset_path
            self.load_hdf5_dataset(verbose=verbose)
        else:
            self.hdf5_dataset_path = None

    def read_data_from_xml(self, datasets_dir='./TEXT_ANNOTATED/', verbose=True):
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

    def create_hdf5_dataset(self, resize=None, file_path='dataset.h5', variable_img_size=True, verbose=True):
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
        else:   # resize
            hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

        # create the dataset in which the image will be stored as flattened arrays
        hdf5_images = hdf5_dataset.create_dataset(name='images', shape=(dataset_size,), maxshape=(None), dtype=h5py.special_dtype(vlen=np.uint8))

        # create the dataset that will hold the image height, widths and channels that we need in order to reconstruct the
        # image from the flattened array later
        hdf5_image_shapes = hdf5_dataset.create_dataset(name='images_shapes', shape=(dataset_size, 3), maxshape=(None, 3),
                                                        dtype=np.int32)

        if self.labels is not None:
            # create the dataset in which the labels will be stored as flattened arrays
            hdf5_labels = hdf5_dataset.create_dataset(name='labels', shape=(dataset_size,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.int32))
            hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes', shape=(dataset_size, 2), maxshape=(None, 2),
                                                            dtype=np.int32)
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

    def load_hdf5_dataset(self, verbose=True):
        """
        Loads an HDF5 dataset that is in the format that "create_hdf5_dataset()' method produces
        :param verbose: (bool, optional) If True, prints out the progress while loading the dataset
        :return: None
        """
        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, "r")
        self.datasets_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.datasets_size)

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['label_shapes']
            label_shapes = self.hdf5_dataset['label_shapes']
            if verbose:
                tr = trange(self.datasets_size, desc="Loading labels", file=sys.stdout)
            else:
                tr = range(self.datasets_size)

            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

    def generate(self, transformation, batch_size=32, shuffle=True, label_encoder=None, returns=None, keep_images_without_gt=False,
                 degenerate_box_handling='remove'):
        """
        Generates batches of samples and corresponding labels indefinitely
        Can shuffle the samples consistently after each complete pass
        :param transformation: (list, optional) A list of transformations that will be applied to the images abd labels
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
        :param degenerate_box_handling: How to handle degenerate boxes, which are boxes that have 'xmax <= xmin' and/or 'ymax <= ymin'.
                                        Degenerate boxes can sometimes be in the dataset, or non-degenerate boxes can become degenerate
                                        after they were processed by transformation
        :return: The next batch as a tuple of items as defined by the 'returns' arguments
        """
        if returns is None:
            returns = {'processed__images', 'encoded_labels'}

        if shuffle:
            object_to_shuffle = [self.dataset_indices]
            if self.filenames is not None:
                object_to_shuffle.append(self.filenames)
            if self.labels is not None:
                object_to_shuffle.append(self.labels)
            shuffled_object = utils.shuffle(*object_to_shuffle)
            for i in range(len(object_to_shuffle)):
                object_to_shuffle[i][:] = shuffled_object

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False, check_min_area=False, check_degenerate=True, labels_format=self.labels_format)

        # Override the labels formats of all the transformation to make







