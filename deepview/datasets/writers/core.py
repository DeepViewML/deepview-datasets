# Copyright 2022 by Au-Zone Technologies.  All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential.
#
# This source code is provided solely for runtime interpretation by Python.
# Modifying or copying any source code is explicitly forbidden.

from deepview.datasets.readers.core import BaseReader
from os.path import exists
from os import makedirs
from typing import Any
from tqdm import tqdm

class BaseWriter(object):
    """
    This class controls the flow of the reader and creates the global behavior of
    an exporter
    """
    def __init__(
        self,
        reader: BaseReader,
        output: str,
        override: bool = False
    ) -> None:
        """
        BaseWriter Class constructor

        Parameters
        ----------
        reader : BaseReader
            Any dataset reader already initialized
        output : str
            Path to the destination folder
        override : bool, optional
            Whether to override the folder or not, by default False
        """
        
        
        if override:
            print(
                f"\t - [WARNING] Output directory is not empty. Exporter will override existing content: {output}"
            )
            makedirs(output, exist_ok=True)
            
        if not exists(output):    
            makedirs(output)
            
        self.__reader__ = reader
        self.__output__ = output
        
    @property
    def output(self):
        """
        Returns the output path where the dataset is going to be stored

        Returns
        -------
        str
            Path to store the dataset
        """
        return self.__output__
    
    
    def export(
        self
    ) -> Any:
        """
        
        THis function returns the iterator associated to a progressbar ready to use for child classes

        Returns
        -------
        Any
            A ``tqdm.tqdm`` iterator
        """
        
        loop = tqdm(
            self.__reader__, 
            desc="\t [INFO] Writing",
            colour="green",
            bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}'
        )
        
        return loop