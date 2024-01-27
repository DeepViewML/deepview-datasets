# Copyright 2022 by Au-Zone Technologies.  All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential.
#
# This source code is provided solely for runtime interpretation by Python.
# Modifying or copying any source code is explicitly forbidden.

from deepview.nn.datasets.iterators.core import BaseIterator
from deepview.nn.datasets.readers import BaseReader
from typing import Any, Iterable
import cv2



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

    def __getitem__(self, item: int) -> Any:
        """
        This function returns the elemnt at position ``item``

        Parameters
        ----------
        item : int
            Instance id

        Returns
        -------
        dict
            A python dictionary with the following structure: {"image": np.ndarray, "boxes": np.ndarray}
        """
        instance = super().__getitem__(item)
        
        instance = {
            "image": cv2.resize(instance[0], (self.__shape__[1], self.__shape__[0])),
            "boxes": instance[1]
        }

        return instance



class TFObjectDetectionIterator(BaseIterator):
    """
    This class represents an Object Detector iterator with optimized 
    pipeline for training models. The core library is TensorFlow
    """
    def __init__(
        self, 
        reader: BaseReader, 
        shape: Iterable, 
        batch_size: int,
        shuffle: bool = False,
        cache: str = None
    ) -> None:
        super().__init__(reader, shape, shuffle)

        self.__cache__ = cache
        if batch_size < 0:
            raise ValueError(
                f"Batch size  smaller than 0 is not alloed: {batch_size} as provided"
            )
        
        self.__batch_size__ = batch_size
        self.__num_batches__ = len(self) //  batch_size if batch_size > 1 else len(self)

        """
        Class constructor

        Parameters
        ----------
        reader : deepview.nn.datasets.reader.BaseReader
            An instance of a dataset reader
        shape : Iterable
            Any iterable in the form (height, width, channels)
        batch_size : int
            Number of elements per batch. If the value is 0, iterator will return elements instead of batches
        shuffle :  bool, optional
            Whether to shuffle or not dataset
        cache : str, optional
            Whether to use a cache on file or not. If cache is a path, then TensorFlow will use it for 
            storing metadata. Otherwise, cache is going to be in memory. In case the dataset is larger
            than memory, TensorFlow will interrupt the training and raise and Error.  
            
            Note: Make sure the application has write permissions on cache folder
        Raises
        ------
        ValueError
            In case the reader is none or unsupported
        ValueError
            In case shape is invalid or None
        ValueError
            In case ``batch_size < 0``
        """
    @property
    def batch_size(self):
        """
        Returns the batch-size
        """
        return self.__batch_size__
    
    @property
    def num_batches(self):
        """
        returns the number of batches in the dataset

        """
        return self.__num_batches__

    @property
    def cache(self) -> str:
        """
        Property that enables the safety reading of cache attribute
        """
        return self.__cache__

    @tf.function
    def get_item(self, item):
        """
        This function wraps a python function (``__getitem__``) in eager execution

        Parameters
        ----------
        item : tf.Tensor
            A tensor representation of item

        Returns
        -------
        tuple
            A tuple containing the resized image and the bounding boxes
        """
        image, boxes = tf.py_function(
            self.__getitem__,
            inp=[item],
            Tout=(tf.uint8, tf.float32)
        )
        image.set_shape(tf.TensorShape(self.__shape__))
        image = tf.image.resize(image, [320, 320])

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
            A batched dataset wrapped into a dictionary format: ``{"images": images, "boxes": boxes}``

        """
        
        
        tf_ids = tf.ragged.constant(self.__annotation_ids__)

        ds_iter = tf.data.Dataset.from_tensor_slices(tf_ids)
        
        if self.__shuffle__:
            ds_iter = ds_iter.shuffle(
                ds_iter.cardinality(), 
                reshuffle_each_iteration=True
            )
        
        ds_iter = ds_iter.map(
            self.get_item, 
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            lambda x, y: {
                "images": x, 
                "boxes": y
            },
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if self.cache is not None:
            ds_iter = ds_iter.cache(
                self.cache
            )
        else:
            ds_iter = ds_iter.cache()

        if self.batch_size > 1:
            ds_iter = ds_iter.padded_batch(
                self.batch_size,
                padded_shapes={
                    "images": self.__shape__,
                    "boxes": [None, 5]
                },
                drop_remainder=True
            )

        return ds_iter

