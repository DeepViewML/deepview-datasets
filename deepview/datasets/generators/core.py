# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from typing import Any
import random
from deepview.datasets.readers import BaseReader


class BaseGenerator(object):
    """
    BaseGenerator This class describes the basic behavior of any Generator
    """

    def __init__(
        self,
        reader:     BaseReader,
        shuffle:    bool = False,
        with_rgb: bool = True,
        with_radar: bool = False,
        with_shapes: bool = False,
        radar_extension: str = 'npy'
    ) -> None:
        """
        Class constructor

        Parameters
        ----------
        reader : deepview.datasets.reader.BaseReader
            An instance of a dataset reader
        shuffle :  bool, optional
            Whether to shuffle or not dataset

        Raises
        ------
        ValueError
            In case the reader is none or unsupported
        ValueError
            In case shape is invalid or None
        """

        if reader is None:
            raise ValueError(
                "``None`` reader was provided"
            )
        self.__reader__ = reader
        self.__shuffle__ = shuffle
        self.__size__ = len(self.__reader__)
        self.__annotation_ids__ = list(range(self.__size__))
        self.__current__ = 0

        self.__use_rgb__ = with_rgb
        self.__use_radar__ = with_radar
        self.__radar_extension__ = radar_extension
        self.__use_shapes__ = with_shapes

        if shuffle:
            random.shuffle(self.__annotation_ids__)

    @property
    def reader(self):
        """
        reader Property overload for safety access to the reader

        Returns
        -------
        reader
            deepview.datasets.reader.BaseReader
        """
        return self.__reader__

    def __getitem__(
        self,
        item: int
    ) -> Any:
        """
        This function returns ``reader[item]``

        Parameters
        ----------
        item : int
            Item id

        Returns
        -------
        tuple
            A two elements tuple containing data used for model input in the
            first position and labels in the second one
        """
        return self.__reader__[item]

    def __len__(self) -> int:
        """
        Returns the number of elements in the iterator

        Returns
        -------
        int
            Number of elements in the iterator. Non batched elements
        """
        return len(self.__reader__)

    def __next__(self) -> Any:
        """
        Returns the next element  by calling ``__getitem__`` function
        """
        if self.__current__ >= self.__size__:
            if self.__shuffle__:
                random.shuffle(self.__annotation_ids__)

            raise StopIteration

        element = self[self.__current__]
        self.__current__ += 1

        return element

    def iterator(self) -> Any:
        """
        This function returns a copy of the current object

        Returns
        -------
        BaseIterator
            A copy of the object
        """
        return self
