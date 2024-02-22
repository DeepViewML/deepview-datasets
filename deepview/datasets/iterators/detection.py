# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from deepview.datasets.iterators.core import BaseIterator
from deepview.datasets.readers import BaseReader
from typing import Any, Iterable
import yaml

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
except ImportError:
    pass


class ObjectDetectionIterator(BaseIterator):
    """
    Abstract class for Object Detection dataset iterators
    """

    def __getitem__(self, item: int) -> tuple:
        """
        This function returns the elemnt at position ``item``

        Parameters
        ----------
        item : int
            Instance id

        Returns
        -------
        tuple
            A tuple containing all the elements from the same instance
        """
        return super().__getitem__(item)


class TFObjectDetectionIterator(BaseIterator):
    """
    This class represents an Object Detector iterator with optimized
    pipeline for training models. The core library is TensorFlow
    """

    def __init__(
        self,
        reader: BaseReader,
        shape: Iterable,
        shuffle: bool = False,
        cache: str = None
    ) -> None:
        """
        Class constructor

        Parameters
        ----------
        reader : deepview.datasets.reader.BaseReader
            An instance of a dataset reader
        shape : Iterable
            Any iterable in the form (height, width, channels)
        shuffle :  bool, optional
            Whether to shuffle or not dataset
        cache : str, optional
            Whether to use a cache on file or not. If cache is a path, then
            TensorFlow will use it for storing metadata. Otherwise, cache is
            going to be in memory. In case the dataset is larger than memory,
            TensorFlow will interrupt the training and raise and Error.

            Note: Make sure the application has write permissions on cache dir
        Raises
        ------
        ValueError
            In case the reader is none or unsupported
        ValueError
            In case shape is invalid or None
        ValueError
            In case ``batch_size < 0``
        """

        super().__init__(reader, shape, shuffle)
        self.__cache__ = cache

    def get_boxes_dimensions(self) -> list:
        return self.reader.get_boxes_dimensions()

    @property
    def cache(self) -> str:
        """
        Property that enables the safety reading of cache attribute
        """
        return self.__cache__

    def __getitem__(self, item: int) -> Any:
        image, boxes = super().reader[item]
        image.set_shape(self.shape)
        image = tf.image.resize(image, (self.height, self.width))
        return image, boxes

    def iterator(
        self
    ) -> Any:
        """
        This function returns a tf.data.Dataset iterator in batch model.
        The reason why elements are returned in batches is because
        according to TensorFlow documentation, Vectorial mapping guarantees
        the higher performance on GPU

        Returns
        -------
        tf.data.Dataset
            A batched dataset wrapped into a dictionary format:
                ``{"images": images, "boxes": boxes}``

        """

        print(self.__annotation_ids__)

        ds_iter = tf.data.Dataset.from_tensor_slices(self.__annotation_ids__)

        if self.cache is not None:
            ds_iter = ds_iter.cache(
                self.cache
            )
        else:
            ds_iter = ds_iter.cache()

        if self.__shuffle__:
            ds_iter = ds_iter.shuffle(
                ds_iter.cardinality(),
                reshuffle_each_iteration=True
            )

        ds_iter = ds_iter.map(
            self.__getitem__,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return ds_iter


class TFObjectDetectionDataset:
    def __init__(
        self,
        config: dict = None
    ) -> None:
        if config is None:
            raise RuntimeError("Empty configuration file was provided")
        
        self.__config__ = config
        self.__train_images__ = None
        self.__train_annotations__ = None

        self.__val_images__ = None
        self.__val_annotations__ = None
        self.__classes__ = None

        self.__train_iterator__ = None
        self.__val_iterator__ = None

        self.load_from_config()

    @property
    def classes(self) -> Iterable:
        return self.__classes__
    
    @property
    def config(self) -> dict:
        return self.__config__

    def load_from_config(self):
        classes = self.__config__.get("classes", None])
        if classes:
            # load mpk 3.0 dataset for training and validation
            self.__classes__ = classes
            return

        from_file = self.__config__.get("config-file", None)
        config = None
        if from_file and os.path.exists(from_file):
            with open(from_file, 'r') as fp:
                config = yaml.safe_load(fp)
        
        if config is None:
            raise ValueError(
                "``config-file`` is missing from dataset definition or file does not exist"
            )
        
        classes = config.get("classes", None)            
        if classes:
            # Load dataset from a config file related with MPK 2.x
            self.__classes__ = classes
            train = config.get("train", None)
            val = config.get("validation", None)

            if train:
                train_images = train.get("images", None)
                train_annotations = train.get("annotations", None)
                if train_images and os.path.exists(train_images):
                    self.__train_images__ = train_images
                else:
                    print(
                        f"\t - [WARNING] Path to training images was not found: {train_images}"
                    )
                
                if train_annotations and os.path.exists(train_annotations):
                    self.__train_annotations__ = train_annotations
                else:
                    print(
                        f"\t - [WARNING] Path to training annotations was not found: {train_annotations}"
                    )

            else:
                print(
                    "\t - [WARNING] No training section was provided. Reade ModelPack 2.x dataset documentation"
                )
            


            return
        
        classes = config.get("names", None)
        if classes:
            # load from yolov-x x > 4 family yaml file
            return 


    def get_train_iterator(self) -> tf.data.Dataset:
        pass

    def get_val_iterator(self) -> tf.data.Dataset:
        pass
