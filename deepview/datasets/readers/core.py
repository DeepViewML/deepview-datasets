# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from typing import Union, Iterable


class BaseReader(Iterable):
    """

    This class represent the abstract class for readers.

    """

    def __init__(
        self,
        classes: Union[str, Iterable],
        silent: bool = False
    ) -> None:
        """
        Class constructot


        Parameters
        ----------
        classes:  Union[str, Iterable]
            Either of a list containing the name of the classes or the path to
            a file containing the classes
        silent : bool, optional
            Whether printing to the console or not, by default False
        """

        self.silent = silent
        self.__storage__ = []
        self.__current__ = -1
        self.__size__ = 0
        self.__classes__ = []

        if isinstance(classes, str):
            with open(classes, 'r') as fp:
                self.__classes__ = [cls.rstrip() for cls in fp.readlines()]
        else:
            self.__classes__ = classes

    @property
    def classes(self):
        """
        Property that overwrites the safe accessing to class member access

        Returns
        -------
        list
            A list of strings containing the name of the classes
        """
        return self.__classes__

    @property
    def storage(self) -> list:
        """
        Property overwrite for safe accessing to class member access

        Returns
        -------
        list
            List that contains all the instances. Values within the list will
            be dataset specific
        """
        return self.__storage__

    def __len__(self) -> int:
        """
        Computes the number of elements within the dataset

        Returns
        -------
        int
            _description_
        """
        return len(self.__storage__)

    def __getitem__(
        self,
        item: int
    ) -> tuple:
        """
        Returns the instance at ``item`` index. At this level the function only
        knows file names. Child classes should be able to override this
        function and return data from files

        Parameters
        ----------
        item : int
            Id of the instance to be retrieved from ``storage``

        Returns
        --------
        tuple
            A tuple containing all the files that represent a single instance

        """

        return self.__storage__[item]

    def __next__(self):
        """
        Returns the next element  by calling ``__getitem__`` function
        """
        if self.__current__ >= self.__size__:
            raise StopIteration

        element = self[self.__current__]
        self.__current__ += 1

        return element

    def __iter__(self):
        """Returns a reference to ``self`` object

        Returns
        -------
        BaseReader
            A copy of current object
        """
        return self
