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


class TFBaseObjectDetectionIterator(BaseIterator):
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


class TFObjectDetectionIterator:
    def __init__(
        self,
        from_config: str,
        shape: Iterable,
        cache: str = None
    ) -> None:
        """Class constructor

        Parameters
        ----------
        from_config : str
           Path to a yaml file containing the dataset information

        shape : Iterable
           Shape to resize the images and boxes. Usually a tuple containing three values (height, width, channels)

        cache : str, optional
            Whether to use a cache on file or not. If cache is a path, then
            TensorFlow will use it for storing metadata. Otherwise, cache is
            going to be in memory. In case the dataset is larger than memory,
            TensorFlow will interrupt the training and raise and Error.

            Note: Make sure the application has write permissions on cache dir

        Raises
        ------
        ValueError
            In case the file provided is not a yaml file
        FileNotFoundError
            In case the location of the file is not found
        """

        self.config = from_config
        self.shape = shape
        self.cache = cache

        if os.path.isfile(from_config) and not from_config.endswith(".yaml"):
            raise ValueError(
                f"Configuration file specified at `from_config` has to be a yaml file: {from_config}"
            )

        if os.path.isfile(from_config) and not os.path.exists(from_config):
            raise FileNotFoundError(
                f"Configuration file provided at `frmo_config` parameter does not exist: {from_config}"
            )

        import yaml
        with open(from_config, 'r') as fp:
            self.config = yaml.safe_load(fp)
        self.config_absolute_path = os.path.dirname(
            os.path.abspath(from_config))

        self.dataset_format = None
        self.load_reader = None

        self.__classes__ = self.load_classes()
        self.training_reader = None

    @property
    def classes(self) -> Iterable:
        """Property wraps for safety access to private attributes

        Returns
        -------
        Iterable
            An iterable containing the name of the classes
        """
        return self.__classes__

    def load_classes(self) -> Iterable:
        """
        This function reads class names from the configuration attribute.
        Inside the function, also the `self.dataset_format` attribute is set to either of
        `modelpack-2.x` or `modelpack-3.x` or `ultralytics`

        Returns
        -------
        Iterable
            A list or tuple containing the classes

        Raises
        ------
        RuntimeError
            If dataset format is not recognized
        """
        classes = self.config.get(
            "classes", []
        )  # loading ModelPack 2.x dataset format
        if len(classes) > 0:
            self.dataset_format = "modelpack-2.x"
            self.load_reader = self.__storage_from_modelpack__
            return classes

        classes = self.config.get("names", [])
        if len(classes) > 0:
            self.dataset_format = "ultralytics"
            self.load_reader = self.__storage_from_ultralytics__
            return classes

        raise RuntimeError(
            "Dataset format was not autodetected. It is does not follow neither\
                of supported formats: ModelPack 2.x, ModelPack 3.x or Ultralytics"
        )

    def __storage_from_modelpack__(self, is_train: bool = True) -> Iterable:

        dataset = self.config.get(
            "train", None) if is_train else self.config.get("validation", None)
        if dataset is None:
            return None

        images = dataset.get("images", None)
        annotations = dataset.get("annotations", None)

        images = os.path.join(self.config_absolute_path, images)
        annotations = os.path.join(self.config_absolute_path, annotations)

        if images.endswith("*.arrow") and annotations.endswith("*.arrow"):
            from deepview.datasets.readers import TFPolarsDetectionReader
            return TFPolarsDetectionReader(
                inputs=images,
                annotations=annotations,
                classes=self.__classes__,
                silent=True
            )
        else:
            from deepview.datasets.readers import TFDarknetDetectionReader
            reader = TFDarknetDetectionReader(
                images=images,
                annotations=annotations,
                classes=self.__classes__,
                silent=True
            )
        return reader

    def __storage_from_ultralytics__(self, is_train: bool = True) -> Iterable:
        from deepview.datasets.readers import TFUltralyticsDetectionReader
        from deepview.datasets.readers import TFDarknetDetectionReader

        dataset = self.config.get(
            "train", None) if is_train else self.config.get("val", None)
        if dataset is None:
            return None

        path = self.config.get("path", None)
        if dataset.endswith(".txt"):
            return TFUltralyticsDetectionReader(
                images=dataset,
                classes=self.__classes__,
                path=path
            )
        else:
            basename = os.path.basename(self.config_absolute_path)
            if dataset.startswith(basename):
                self.config_absolute_path = os.path.dirname(
                    self.config_absolute_path)

            images = os.path.join(self.config_absolute_path, dataset)
            annotations = os.path.normpath(images).split(os.path.sep)
            annotations[-2] = "labels"
            if annotations[0] == '':
                annotations[0] = os.path.sep
            annotations = os.path.join(*annotations)
            annotations = annotations.replace(":", ":\\")

            return TFDarknetDetectionReader(
                images=images,
                annotations=annotations,
                classes=self.__classes__,
                silent=True
            )

    def __get_iterator__(self, is_train: bool = True) -> BaseIterator:
        """This function returns the Iterator instance according to the dataset format

        Parameters
        ----------
        is_train : bool, optional
            Whether the dataset partition is loaded from train or validatoin samples, by default True

        Raises
        ------
        RuntimeError
            If dataset format in case the dataset does not exists or it is empty

        Returns
        -------
        BaseIterator
            The iterator ready to be consumed by the training iterators. Note: data is not batched or augmented at this point
        """
        reader = self.load_reader(is_train=is_train)
        if is_train:
            self.training_reader = reader
        else:
            self.val_reader = reader
        return TFBaseObjectDetectionIterator(
            reader=reader,
            shape=self.shape,
            shuffle=is_train,
            cache=self.cache if is_train else None
        )

    
    
    def get_train_iterator(self) -> BaseIterator:
        """This function creates the Train iterator and return it

        Returns
        -------
        BaseIterator
            An iterator loaded from training partition.

        Raises
        --------
        RuntimeError
            In case the training iterator is None. Training set is mandatory
        """
        train_handler = self.__get_iterator__(is_train=True)

        if train_handler is None:
            raise RuntimeError(
                "Training dataset was not properly loaded from source."
            )

        return train_handler

    def get_val_iterator(self) -> BaseIterator:
        """This function creates the Validation iterator and return it

        Returns
        -------
        BaseIterator
            An iterator loaded from validation partition.
        """
        return self.__get_iterator__(is_train=False)

    def get_boxes_dimensions(self) -> Iterable:
        """This function returns all the pairs width,height from bounding boxes in the training 
        set to compute anchors

        Returns
        -------
        Iterable
            The Iterable instance containing pairs of width,height
        """
        return self.training_reader.get_boxes_dimensions()