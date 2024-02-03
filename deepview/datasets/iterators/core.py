# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from deepview.datasets.readers import BaseReader
from typing import Iterable, Any
import random


class BaseIterator(object):
    def __init__(
        self,
        reader:     BaseReader,
        shape:      Iterable,
        shuffle:    bool = False
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
        if shape is None or len(shape) != 3:
            raise ValueError(
                f"Unsupported shape was provided: {shape}"
            )
        self.__reader__ = reader
        self.__shape__ = shape
        self.__shuffle__ = shuffle
        self.__annotation_ids__ = list(range(len(self.__reader__)))
        self.__current__ = 0
        self.__size__ = len(self.__reader__)

        if shuffle:
            random.shuffle(self.__annotation_ids__)

    @property
    def shape(self):
        return self.__shape__

    @property
    def height(self) -> int:
        return self.__shape__[0]

    @property
    def width(self) -> int:
        return self.__shape__[1]

    @property
    def channels(self) -> int:
        return self.__shape__[2]

    @property
    def reader(self):
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
