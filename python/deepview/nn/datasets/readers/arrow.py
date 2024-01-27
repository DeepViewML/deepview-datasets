# Copyright 2022 by Au-Zone Technologies.  All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential.
#
# This source code is provided solely for runtime interpretation by Python.
# Modifying or copying any source code is explicitly forbidden.

from deepview.nn.datasets.readers.core import BaseReader
from typing import Iterable
import polars as pl
import numpy as np


class PolarsDetectionReader(BaseReader):
    """
    This class wraps polars library to efficiently load a dataset
    """
    
    def __init__(
        self,
        inputs: str,
        annotations: str,
        silent: bool = False,
        classes: Iterable = None
    ) -> None:
        super().__init__(
            classes=[],
            silent=silent
        )
        """
        Class constructor

        Parameters
        -----------
        inputs : str
            Path containing the model input data. For example, in a detection model, 
            the inputs are going to be the path to ``images_*.arrow``
            
        
        annotations : Union[str, Iterable]
            Either of the path to the folder containng *.txt files or a list 
            containing the path to all the 
        
        classes:  Union[str, Iterable]
            Either of a list containing the name of the classes or the path to a file containing the classes
            
        silent : bool, optional, default False
             Whether printing to the console or not, by default False

        Raises
        ------
        FileNotFoundError
            An exception is thrown in case path to images or annotations does not exist
        
        Return
        ------
        None
        
        """
        
        self.__inputs__ = pl.scan_ipc(
            inputs
        )
        
        self.__annotations__ = pl.scan_ipc(
            annotations
        )
        if classes is None:
            self.__classes__ = self.__annotations__.select(pl.col("class")).unique().collect()['class'].to_list()
            self.__annotations_ids__ = self.__inputs__.select(pl.col("id")).collect()["id"].to_list()
        else:
            self.__classes__ = classes
            self.__annotations_ids__ = self.__annotations__.filter(
                    pl.col("class").is_in(classes)
                ).select(pl.col("id")).collect().unique()["id"].to_list()
        
        
        self.__size__ = len(self.__annotations_ids__)
        self.__current__ = 0
        
    def __len__(self) -> int:
        return self.__size__
    
    @property
    def classes(self):
        return self.__classes__

    
    def __getitem__(
        self, 
        item: int
    ) -> tuple:
        instance_id = self.__annotations_ids__[item]
        
        image, shape = self.__inputs__.filter(pl.col("id").eq(instance_id)).select([pl.col("data"), pl.col("shape")]).collect()
        image = np.asarray(image.item(), dtype=np.uint8).reshape(shape.item(0)).copy()
        
        bboxes, classes = self.__annotations__.filter(pl.col("id").eq(instance_id)).select([pl.col("box2d"), pl.col("class")]).collect()
        bboxes = np.asarray(bboxes.to_list(), dtype=np.float32)
        classes = np.asarray(classes.cast(pl.Categorical).to_physical(), dtype=np.int32)
        classes = classes[:, None]
        
        if len(bboxes) == 0:
            return image, np.zeros(shape=(1, 5), dtype=np.float32)
        
        boxes = np.concatenate([bboxes, classes], axis=1)
        return image, boxes

    