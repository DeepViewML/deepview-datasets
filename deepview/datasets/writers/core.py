# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from deepview.datasets.readers.core import BaseReader
from os.path import exists
from os import makedirs
from typing import Any


class BaseWriter(object):
    """
    This class controls the flow of the reader and creates the global behavior
    of an exporter
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

        THis function returns the iterator associated to a progressbar ready to
        use for child classes

        Returns
        -------
        Any
            A reader
        """

        return self.__reader__

    def export_dataset_configuration_file(
        self, 
        file: str,
        train_set: str,
        val_set: str
    ) -> None:
        "This function saves the yaml file into disk"
        raise NotImplementedError("Abstract method should be implemented in child classes")