# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from deepview.datasets.iterators.core import BaseIterator
from deepview.datasets.readers import BaseReader
from typing import Any, Iterable


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
