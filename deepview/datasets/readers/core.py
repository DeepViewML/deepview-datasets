# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from typing import Union, Iterable
import random
import numpy as np
import yaml


class BaseReader(Iterable):
    """

    This class represent the abstract class for readers.

    """

    def __init__(
        self,
        classes: Union[str, Iterable],
        silent: bool = False,
        shuffle: bool = False
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
        shuffle : bool, optional
            This parameter force data to be shuffled everytime the iterator ends, Default to False            
        """

        self.silent = silent
        self.__storage__ = []
        self.__current__ = -1
        self.__size__ = 0
        self.__classes__ = []
        self.__instance_id__ = None
        self.__shuffle__ = shuffle

        if isinstance(classes, str):
            if classes.endswith(".txt"):
                with open(classes, 'r', encoding='utf-8') as fp:
                    self.__classes__ = [cls.rstrip() for cls in fp.readlines()]
            elif classes.endswith('.yaml'):
                with open(classes, 'r', encoding='utf-8') as fp:
                    metadata = yaml.safe_load(fp)
                    self.__classes__ = metadata.get('classes', [])
                    if len(self.__classes__) == 0:
                        self.__classes__ = metadata.get('names', [])
                        if isinstance(self.__classes__, dict):
                            self.__classes__ = [
                                value for _, value in self.__classes__.items()
                            ]

                    if len(self.__classes__) == 0:
                        raise ValueError(
                            f"No reference to class names was found in {classes}"
                        )
            else:
                raise ValueError(
                    "Unsupported file format was given for classes. Either of yaml or txt file"
                )
        else:
            self.__classes__ = classes

    def get_instance_id(self):
        """
        get_instance_id This functoin 

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        RuntimeWarning
            _description_
        """
        if self.__instance_id__ is None:
            raise RuntimeWarning(
                "self.__instance_id__ is None. Bad property initialization"
            )
        return self.__instance_id__

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
        if self.__shuffle__ and item < 1:
            random.shuffle(self.__storage__)

        return self.__storage__[item]

    def __next__(self):
        """
        Returns the next element  by calling ``__getitem__`` function
        """
        if self.__current__ >= self.__size__:
            self.__current__ = 0
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

    def random(self):
        """This function returns a random element from the dataset

        Returns
        -------
        tuple
            Contains the inputs and labels for a given random element in the dataset

        """
        item = np.random.randint(0, self.__size__ - 1)
        return self[item]


class ObjectDetectionBaseReader(BaseReader):
    """This class wraps the Object Detection Dataset Reader

    Parameters
    ----------
    BaseReader : Base class
        Base class for all the readers
    """

    def get_boxes_dimensions(self) -> Iterable:
        """This function extracts bounding boxes dimensions for anchors computation purposes

        Returns
        -------
        Iterable
            An np.ndarray of shape (N, 2) composed by all the bounding boxes (width, height)

        Raises
        ------
        NotImplementedError
            Abstract method
        """
        raise NotImplementedError("Abstract method")
