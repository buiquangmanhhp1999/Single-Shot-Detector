from config import subtract_mean, img_height, img_width, scales_df
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from layers import ssd_300
from loss import SSDLoss
from DataGenerator import DataGenerator
from ssd_encoder.ssd_input_encoder import SSDInputEncoder
from augmentation.augmentation import SSDDataAugmentation
from augmentation.object_detection_2d_geometric_ops import Resize
from augmentation.object_detection_2d_photometric_ops import ConvertTo3Channels
import math
import os


# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch < 30:
        return 0.01
    elif epoch < 100:
        return 0.001
    else:
        return 0.0001


class SSDTrain:
    def __init__(self, weight_path=None, two_boxes_for_ar1=True, clip_boxes=False, normalize_coords=True):
        if weight_path is not None:
            self.weight_path = weight_path
        self.batch_size = 16
        self.epochs = 30
        # 1. Build the Keras model
        K.clear_session()  # clear previous models from memory
        self.model = ssd_300(mode='train', l2_reg=0.005, two_boxes_for_ar1=two_boxes_for_ar1, clip_boxes=clip_boxes,
                             normalize_coords=normalize_coords)
        # self.model.summary()
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
        ssd_loss = SSDLoss()
        self.model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

        # load dataset
        train_dataset = DataGenerator(trainable=True, load_images_into_memory=False, filenames='./data/train_img_path.txt')
        val_dataset = DataGenerator(trainable=False, load_images_into_memory=False, filenames='./data/val_img_path.txt')

        train_dataset.read_data_from_xml(datasets_dir='./data/train/')
        val_dataset.read_data_from_xml(datasets_dir='./data/validate/')
        # convert the dataset into hdf5 dataset. This will require more disk space, but will speed up the training.
        # Doing this is not relevant in case you activated the 'load_images_into_memory'.
        if not os.path.isfile('./data/train_val_datasets.h5'):
            train_dataset.create_hdf5_dataset(resize=False, file_path='./data/train_val_datasets.h5', variable_img_size=True,
                                              verbose=True)
            val_dataset.create_hdf5_dataset(resize=False, file_path='./data/test_datasets.h5')
        else:
            train_dataset.load_hdf5_dataset(verbose=True, file_path='./data/train_val_datasets.h5')
            val_dataset.load_hdf5_dataset(verbose=True, file_path='./data/test_datasets.h5')

        # set the image transformations for pre-processing and data augmentation options
        ssd_data_augmentation = SSDDataAugmentation(img_height=img_height, img_width=img_width, background=subtract_mean)

        # The encode constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes
        predictor_sizes = [self.model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                           self.model.get_layer('fc7_mbox_conf').output_shape[1:3],
                           self.model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                           self.model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                           self.model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                           self.model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

        ssd_input_encoder = SSDInputEncoder(predictor_sizes=predictor_sizes, scales=scales_df, clip_boxes=clip_boxes,
                                            two_boxes_for_ar1=two_boxes_for_ar1,
                                            pos_iou_threshold=0.5, neg_iou_limit=0.5, normalize_coords=normalize_coords)

        # For the validation generator:
        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=img_height, width=img_width)

        # Create the generator handles that will be passed to Keras 'fit' function
        self.train_generator = train_dataset.generate(batch_size=self.batch_size, shuffle=True,
                                                      transformations=[ssd_data_augmentation],
                                                      label_encoder=ssd_input_encoder,
                                                      returns={'processed_images', 'encoded_labels'},
                                                      keep_images_without_gt=False)
        self.val_generator = val_dataset.generate(batch_size=self.batch_size, shuffle=False,
                                                  transformations=[convert_to_3_channels, resize],
                                                  label_encoder=ssd_input_encoder,
                                                  returns={'processed_images', 'encoded_labels'},
                                                  keep_images_without_gt=False)
        # Get the number of samples in the training datasets
        self.train_dataset_size = train_dataset.get_dataset_size()
        self.val_dataset_size = val_dataset.get_dataset_size()
        print('Number of images in the training datasets:\t{:>6}'.format(self.train_dataset_size))
        print("Number of images in the validation dataset:\t{:>6}".format(self.val_dataset_size))

    def train(self):
        model_checkpoint = ModelCheckpoint(
            filepath='ssd300_epoch-{epoch:02d}.h5',
            monitor='val_loss', verbose=1, save_weights_only=True)
        learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
        terminate_on_nan = TerminateOnNaN()
        csv_logger = CSVLogger(filename='ssd300_training_log.csv', separator=',', append=True)
        callbacks = [model_checkpoint, csv_logger, learning_rate_scheduler, terminate_on_nan]

        self.model.fit(x=self.train_generator, steps_per_epoch=1000, epochs=self.epochs,
                       callbacks=callbacks, validation_data=self.val_generator, shuffle=True,
                       validation_steps=math.ceil(self.val_dataset_size / self.batch_size) * 2)
