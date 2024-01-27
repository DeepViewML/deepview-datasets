# Copyright 2022 by Au-Zone Technologies.  All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential.
#
# This source code is provided solely for runtime interpretation by Python.
# Modifying or copying any source code is explicitly forbidden.


from deepview.nn.datasets.readers import BaseReader
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
        reader : deepview.nn.datasets.reader.BaseReader
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
        if shape is None or len(shape) !=3:
            raise ValueError(
                f"Unsupported shape was provided: {shape}"
            )
        self.__reader__         = reader
        self.__shape__          = shape
        self.__shuffle__        = shuffle
        self.__annotation_ids__ = list(range(len(self.__reader__)))
        self.__current__        = 0
        self.__size__           = len(self.__reader__)

        if shuffle:
            random.shuffle(self.__annotation_ids__)

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
            A two elements tuple containing data used for model input in the first 
            position and labels in the second one
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