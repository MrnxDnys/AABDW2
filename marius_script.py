import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import segmentation_models as sm
import albumentations as A


# Script for training of a segmentation model
# Adapted from https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        start_index (int): index for ids where this dataset should start (inclusive)
        end_index (int): index for ids where this dataset should end (exclusive)
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            json_file,
            start_index,
            end_index,
            augmentation=None,
            preprocessing=None,
            ):
        self.ids = os.listdir(images_dir)[start_index:end_index]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros_like(image)

        # extract bounding polygon from json
        bounds = self.data[f"{self.ids[i]}"]["bounds_x_y"]
        xs = []
        ys = []
        for x_y in bounds:
            xs.append(x_y["x"])
            ys.append(x_y["y"])

        # fill in bounding polygon
        contours = np.rint(np.array([xs, ys])).astype(int).T
        cv2.fillPoly(mask, pts=[contours], color=1)
        mask = mask[..., [0]].astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, shift_limit=0.1, p=1),

        # A.RandomCrop(height=256, width=256, always_apply=True),
        A.RandomSizedCrop((128, 384), 256, 256, always_apply=True),
        A.PadIfNeeded(min_height=None, min_width=None,
                      pad_height_divisor=32, pad_width_divisor=32),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
                ],
            p=0.9,
            ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
                ],
            p=0.9,
            ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
                ],
            p=0.9,
            ),
        # A.Lambda(mask=round_clip_0_1)
        ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(min_height=None, min_width=None,
                      pad_height_divisor=32, pad_width_divisor=32),
        ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        ]
    return A.Compose(_transform)


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def train_model(train_dataset, valid_dataset, checkpoint, BACKBONE='efficientnetb3'):
    sm.set_framework('tf.keras')

    BATCH_SIZE = 6
    LR = 0.0001
    EPOCHS = 5
    model = sm.Unet(BACKBONE, classes=1,weights=checkpoint, activation="sigmoid")

    # define optimizer
    optim = tf.keras.optimizers.Adam(LR)

    total_loss = sm.losses.binary_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimizer, loss and metrics
    model.compile(optim, total_loss, metrics)
    # if checkpoint is not None:
    #     model.load_weights(checkpoint)


    train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = Dataloader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best_model.h5",
                                           save_weights_only=True, save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(),
        ]

    # train model
    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=val_dataloader,
        validation_steps=len(val_dataloader),
        )
    return model, history


if __name__ == '__main__':
    # TODO: change based on your file path
    BACKBONE = 'efficientnetb3'

    loc_base = "."
    loc_images = os.path.join(loc_base, "images")
    loc_meta = os.path.join(loc_base, "metadata.json")
    dataset_size = len(next(os.walk(loc_images))[2])
    preprocess_input = sm.get_preprocessing(BACKBONE)
    # set sizes of train/valid/test set
    train_end = int(0.7 * dataset_size)
    val_end = int(0.85 * dataset_size)

    # Dataset for train images
    train_dataset = Dataset(loc_images, loc_meta,
                            start_index=0, end_index=train_end,
                            preprocessing=get_preprocessing(preprocess_input),
                            augmentation=get_training_augmentation())

    # Dataset for validation images
    valid_dataset = Dataset(
        loc_images,
        loc_meta,
        start_index=train_end, end_index=val_end,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
        )
    checkpoint = None
    if os.path.exists("best_model.h5"):
        checkpoint = "best_model.h5"
    model, history = train_model(train_dataset, valid_dataset,  checkpoint,BACKBONE)
