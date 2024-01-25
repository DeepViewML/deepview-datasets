# Copyright 2022 by Au-Zone Technologies.  All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential.
#
# This source code is provided solely for runtime interpretation by Python.
# Modifying or copying any source code is explicitly forbidden.

import polars

import polars as pl
from typing import Iterable, Union

class BaseReader(object):
    
    def read_instance(
        self,
        instance: Union[Iterable, str]
    ) -> pl.LazyFrame:
        """
        This function represents acts like a template that defines the way
        the readers will load instances and return them.

        Parameters
        ----------
        instance : Union[Iterable, str]
            It could be either of, the path to the file containing a full  instance sample 
            (it could be multiples samples as well, like tfrecord) or a list taht contains the 
            path to each file that represents an instance. For example, darknet format for object 
            detection encodes instances with two files, image and annotations. A valid input
            for this function could be:
            
            Example 1: Reading instance/instances stored in a single file
                ::

                    object.read_instance(
                        instance="/data/coco/0001.tfrecord"
                    )


            Example 2: Reading instance/instance stored in multiple files
                ::

                    object.read_instance(
                        instance=[
                            "/data/images/0001.jpg",
                            "/data/labels/0001.txt",
                        ]
                    )
            
            Example 3: A more advanced usage for Raivin (*.mcap reader)
                ::

                    object.read_instance(
                        instance=/data/rosbags/0001.mcap
                    )

        Returns
        -------
        pl.LazyFrame
            A Lazy polars DataFrame. See here: https://docs.pola.rs/py-polars/html/reference/lazyframe/index.html

        Raises
        ------
        ValueError
            _description_
        """
        
        raise ValueError(
            "Abstract methods should be implemented on child classes"
        )