from SSD.config import scales, aspect_ratios_per_layer, var, swap_channels, subtract_mean
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from SSD.layers import ssd_300
from SSD.loss import multi_box_loss
from SSD.DataGenerator import DataGenerator


class SSDTrain:
    def __init__(self, mode='training', hdf5_dataset_path=None,  weight_path=None, two_boxes_for_ar1=True, clip_boxes=False,
                 normalize_coords=True):
        if weight_path is not None:
            self.weight_path = weight_path

        # 1. Build the Keras model
        K.clear_session()  # clear previous models from memory
        self.model = ssd_300(mode=mode, l2_reg=0.0005, two_boxes_for_ar1=two_boxes_for_ar1, clip_boxes=clip_boxes,
                             normalize_coords=normalize_coords)
        sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
        self.model.compile(optimizer=sgd, loss=multi_box_loss)

        # load dataset
        train_dir = './TEXT_ANNOTATED/'
        if mode == 'training':
            if hdf5_dataset_path is None:
                train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
                train_dataset.read_data_from_xml(datasets_dir=train_dir)

                # convert the dataset into hdf5 dataset. This will require more disk space, but will speed up the training.
                # Doing this is not relevant in case you activated the 'load_images_into_memory'.

                train_dataset.create_hdf5_dataset(resize=False, file_path='train_val_datasets.h5', variable_img_size=True, verbose=True)
            else:
                train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=hdf5_dataset_path)

        # The encode constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes
        predictor_sizes = [self.model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                           self.model.get_layer('fc7_mbox_conf').output_shape[1:3],
                           self.model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                           self.model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                           self.model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                           self.model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]
        




