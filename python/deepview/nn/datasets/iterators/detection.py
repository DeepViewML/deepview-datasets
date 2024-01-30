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

try:
    import cv2
except ImportError:
    pass

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
        reader : deepview.nn.datasets.reader.BaseReader
            An instance of a dataset reader
        shape : Iterable
            Any iterable in the form (height, width, channels)
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
            A batched dataset wrapped into a dictionary format: ``{"images": images, "boxes": boxes}``

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


if __name__ == '__main__':
    from deepview.nn.datasets.readers import TFPolarsDetectionReader

    reader = TFPolarsDetectionReader(
        inputs = "/home/reinier/development/deepview-datasets/demos/python/playingcards-polars/train/images_*.arrow",
        annotations = "/home/reinier/development/deepview-datasets/demos/python/playingcards-polars/train/boxes_*.arrow"
    )
    
    # Loading dataset cache into memory
    iterator = TFObjectDetectionIterator(
        reader=reader,
        shape=(480, 640, 3),
        cache="/home/reinier/development/deepview-datasets/demos/python/playingcards-polars/tf-cache-2"
    )

    import time

    print("measuring augmentation speed...")
    num_iters = len(iterator)

    st = time.time()
    
    for i, instance in enumerate(iterator.iterator()):
        images = instance[0]
        boxes = instance[1]
        print(images.shape, boxes.shape)
    ed = time.time()
    print(f"{1 / ((ed - st) / num_iters) * 9:.3f} FPS")