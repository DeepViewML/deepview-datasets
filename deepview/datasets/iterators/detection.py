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


class TFObjectDetectionIterator:
    """
    This class defines the functionalities for all custom dataset that loads any reader, auto-discovery reader.
    """

    def __init__(
        self,
        from_config: Any = None
    ) -> None:

        self.config = from_config

        if self.config is None:
            raise ValueError(
                "Dataset can not be loaded when `from_config` parameter is `None`"
            )

        if os.path.isfile(from_config) and not from_config.endswith(".yaml"):
            raise ValueError(
                f"Configuration file specified at `from_config` has to be a yaml file: {from_config}"
            )

        if os.path.isfile(from_config) and not os.path.exists(from_config):
            raise FileNotFoundError(
                f"Configuration file provided at `frmo_config` parameter does not exist: {from_config}"
            )

        if os.path.isfile(from_config):
            import yaml
            with open(from_config, 'r') as fp:
                self.config = yaml.safe_load(fp)

        self.dataset_format = None
        self.__classes__ = self.load_classes()
        

    def load_classes(self) -> Iterable:
        """
        This function reads class names from the configuration attribute
        """
        classes = self.config.get(
            "classes", [])  # loading ModelPack 2.x dataset format
        if len(classes) > 0:
            self.dataset_format = "modelpack-2.x"
            return classes

        # loading ModelPack 3.x dataset format
        dataset = self.config.get("dataset", None)
        classes = dataset.get("classes") if dataset else []
        if len(classes) > 0:
            self.dataset_format = "modelpack-3.x"
            return classes

        classes = self.config.get("names", [])
        if len(classes) > 0:
            self.dataset_format = "ultralytics"
            return classes

        raise RuntimeError(
            "Dataset format was not autodetected. It is does not follow neither\
                of supported formats: ModelPack 2.x, ModelPack 3.x or Ultralytics"
        )
