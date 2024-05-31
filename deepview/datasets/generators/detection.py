# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from typing import Iterable
import io
import os
from PIL import Image
import numpy as np
import yaml
from deepview.datasets.generators.core import BaseGenerator
from deepview.datasets.readers import DarknetDetectionReader
from deepview.datasets.readers import UltralyticsDetectionReader
from deepview.datasets.readers import PolarsDetectionReader


class BaseObjectDetectionGenerator(BaseGenerator):
    """
    Abstract class for Object Detection dataset generator
    """

    def __getitem__(self, item: int) -> list:
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
        instance = list(super().__getitem__(item))
        if self.__use_rgb__:
            image = Image.open(io.BytesIO(instance[0])).convert('RGB')
            image = np.asarray(image, dtype=np.uint8)
            instance[0] = image

        return instance

    def get_boxes_dimensions(self) -> Iterable:
        """
        get_boxes_dimensions returns the list of bounding boxes dimensions from the entire dataset in the way of (width, height)

        Returns
        -------
        Iterable
            Any iterable containing all the dimensions on the dataset
        """
        return self.reader.get_boxes_dimensions()

    def random(self):
        item = np.random.randint(0, len(self.__reader__) - 1)
        return self[item]


class ObjectDetectionGeneratorFromRadar(BaseObjectDetectionGenerator):
    def __getitem__(self, item: int) -> list:
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
        instance = list(super().__getitem__(item))

        return instance

    def get_boxes_dimensions(self) -> Iterable:
        """
        get_boxes_dimensions returns the list of bounding boxes dimensions from the entire dataset in the way of (width, height)

        Returns
        -------
        Iterable
            Any iterable containing all the dimensions on the dataset
        """
        return self.reader.get_boxes_dimensions()

    def random(self):
        item = np.random.randint(0, len(self.__reader__) - 1)
        return self[item]


class ObjectDetectionGenerator:
    def __init__(
        self,
        from_config: str,
        groups: Iterable = None,
        with_rgb: bool = True,
        with_radar: bool = False,
        with_distances: bool = False,
        radar_extension: str = 'npy',
        class_mask: Iterable = None
    ) -> None:
        """Class constructor

        Parameters
        ----------
        from_config : str
           Path to a yaml file containing the dataset information

        Raises
        ------
        ValueError
            In case the file provided is not a yaml file
        FileNotFoundError
            In case the location of the file is not found
        """
        self.__use_rgb__ = with_rgb
        self.__use_radar__ = with_radar
        self.__radar_extension__ = radar_extension
        self.__use_distances__ = with_distances
        self.__class_mask__ = class_mask

        self.config = from_config

        if os.path.isfile(from_config) and not from_config.endswith(".yaml"):
            raise ValueError(
                f"Configuration file specified at `from_config` has to be a yaml file: {from_config}"
            )

        if os.path.isfile(from_config) and not os.path.exists(from_config):
            raise FileNotFoundError(
                f"Configuration file provided at `frmo_config` parameter does not exist: {from_config}"
            )

        with open(from_config, 'r', encoding="utf-8") as fp:
            self.config = yaml.safe_load(fp)

        self.config_absolute_path = os.path.dirname(
            os.path.abspath(from_config))

        self.dataset_format = None
        self.load_reader = None

        self.__classes__ = self.load_classes()
        self.training_reader = None
        self.val_reader = None
        self.groups = groups

    @property
    def classes(self) -> Iterable:
        """Property wraps for safety access to private attributes

        Returns
        -------
        Iterable
            An iterable containing the name of the classes
        """
        if self.training_reader is not None:
            return self.training_reader.classes
        if self.val_reader is not None:
            return self.val_reader.classes
        raise RuntimeError("No reader has been intantiated yet !")

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
            return classes if self.__class_mask__ is None else self.__class_mask__

        classes = self.config.get("names", [])
        if len(classes) > 0:
            self.dataset_format = "ultralytics"
            self.load_reader = self.__storage_from_ultralytics__
            return classes if self.__class_mask__ is None else self.__class_mask__

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
            return PolarsDetectionReader(
                inputs=images,
                annotations=annotations,
                classes=self.__classes__,
                silent=True,
                shuffle=is_train
            )
        else:
            reader = DarknetDetectionReader(
                images=images,
                annotations=annotations,
                classes=self.__classes__,
                silent=True,
                shuffle=is_train,
                groups=self.groups,
                with_radar=self.__use_radar__,
                with_rgb=self.__use_rgb__,
                with_distances=self.__use_distances__,
                radar_extension=self.__radar_extension__
            )
        return reader

    def __storage_from_ultralytics__(self, is_train: bool = True) -> Iterable:

        dataset = self.config.get(
            "train", None) if is_train else self.config.get("val", None)

        if dataset is None:
            return None

        path = self.config.get("path", None)
        if dataset.endswith(".txt"):
            return UltralyticsDetectionReader(
                images=dataset,
                classes=self.__classes__,
                path=path,
                shuffle=is_train,
                groups=self.groups
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

            return DarknetDetectionReader(
                images=images,
                annotations=annotations,
                classes=self.__classes__,
                silent=True,
                shuffle=is_train,
                groups=self.groups,
                with_radar=self.__use_radar__,
                with_rgb=self.__use_rgb__,
                with_distances=self.__use_distances__,
                radar_extension=self.__radar_extension__
            )

    def __get_generator__(self, is_train: bool = True) -> BaseGenerator:
        """This function returns the Generator instance according to the dataset format

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
        BaseGenerator
            The iterator ready to be consumed by the training iterators. Note: data is not batched or augmented at this point
        """
        reader = self.load_reader(is_train=is_train)
        if is_train:
            self.training_reader = reader
        else:
            self.val_reader = reader

        if self.__use_rgb__:
            return BaseObjectDetectionGenerator(
                reader=reader,
                shuffle=is_train,
                with_radar=self.__use_radar__,
                with_rgb=self.__use_rgb__,
                with_distances=self.__use_distances__,
                radar_extension=self.__radar_extension__
            )
        elif self.__use_radar__:
            return ObjectDetectionGeneratorFromRadar(
                reader=reader,
                shuffle=is_train,
                with_radar=self.__use_radar__,
                with_rgb=self.__use_rgb__,
                with_distances=self.__use_distances__,
                radar_extension=self.__radar_extension__
            )

    def get_train_generator(self) -> BaseGenerator:
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
        train_handler = self.__get_generator__(is_train=True)
        if train_handler is None:
            raise RuntimeError(
                "Training dataset was not properly loaded from source."
            )

        return train_handler.iterator()

    def get_val_generator(self) -> BaseGenerator:
        """This function creates the Validation generator and return it

        Returns
        -------
        BaseIterator
            An iterator loaded from validation partition.
        """
        val_handler = self.__get_generator__(is_train=False)
        return val_handler.iterator()

    def get_boxes_dimensions(self, train: bool = True) -> Iterable:
        """This function returns all the pairs width,height from bounding boxes in the training 
        set to compute anchors

        Returns
        -------
        Iterable
            The Iterable instance containing pairs of width,height
        """
        if train:
            reader = self.training_reader
        else:
            reader = self.val_reader

        if reader is None:
            reader = self.__get_generator__(is_train=train)

        return reader.get_boxes_dimensions()

    def get_class_distribution(self, train: bool = True) -> dict:
        """
        This function computes the number of instances per class and return them into a dictionary

        Parameters
        ----------
        trian : bool, optional
            _description_, by default True

        Returns
        -------
        dict
            _description_
        """
        if train:
            reader = self.training_reader
        else:
            reader = self.val_reader

        if reader is None:
            reader = self.__get_generator__(is_train=train)

        return reader.get_class_distribution()
