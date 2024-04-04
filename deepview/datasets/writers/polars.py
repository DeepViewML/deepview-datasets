# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from deepview.datasets.writers.core import BaseWriter
from deepview.datasets.readers.core import BaseReader
from os.path import join
import polars as pl
import numpy as np


class PolarsWriter(BaseWriter):
    def __init__(
        self,
        reader: BaseReader,
        output: str,
        override: bool = False,
        max_file_size: float = 2.
    ) -> None:
        """
        PolarsWriter Class constructor

        Parameters
        ----------
        reader : BaseReader
            Any dataset reader already initialized
        output : str
            Path to the destination folder
        override : bool, optional
           Whether to override the folder or not, by default False
        max_file_size: float, optional
            Maximum size in GB per file to be stored into arrow format, by
            default 2.
        """
        super().__init__(reader, output, override)

        self.__max_file_size__ = max_file_size

    @property
    def max_file_size(self):
        """Returns the maximum size allowed (GB) per dataset chunk. Only taken
        into consideration for polars subclasses

        Returns
        -------
        float
            A value representing an amoung in GB
        """
        return self.__max_file_size__

    def write_instance(self, instance: tuple):
        """
        This function creates a pl.DataFrame from the tuple. The child writer
        class needs to be in agreement with the reader.

        Parameters
        ----------
        instance : tuple
            Any tuple containing the output of reader __getitem__
        """


class PolarsDetectionWriter(PolarsWriter):

    def write_instance(
        self,
        instance: tuple,
        key: str
    ) -> tuple:
        """
        This function creates a pl.DataFrame from a tuple instance. The first
        element on the instance is the image and the second one are boxes. The
        function will create an image pl.DataFrame as well as a pl.DataFrame
        for boxes

        Parameters
        ----------
        instance : tuple
            Any tuple containing the output of reader __getitem__

        key : str
            A value used as primary key to store instance within dataset

        Return
        ------
        tuple
            A tuple containing the image data frame as first position and
            annotations data frame in the second one

        """

        image, annotations = instance
        img = pl.Series("image", [image], dtype=pl.List(pl.UInt8))
        image_data_frame = pl.DataFrame({'id': key, "data": img})

        if len(annotations) == 0:
            return image_data_frame, None

        boxes = annotations[:, [0, 1, 2, 3]]  # reading boxes coordinates
        labels = annotations[:, [4]]         # reading labels

        cat_names = np.asarray(self.__reader__.classes)[
            labels.astype(np.int32)
        ].flatten().tolist()

        with pl.StringCache():
            annotations_data_frame = pl.DataFrame({
                "id": key,
                "class": pl.Series("class", cat_names, dtype=pl.Categorical),
                "box2d": [
                    pl.Series("box2d", b.tolist(), dtype=pl.Float32) for b in boxes
                ],
            })

        return image_data_frame, annotations_data_frame

    def export(self) -> None:
        pl.enable_string_cache()

        df_images = None
        df_annotations = None
        file_size_counter = 1
        instance_id_counter = 1

        max_string_padd = len(str(len(self.__reader__)))

        loop = super().export()

        for instance in loop:
            key = '{instance_id:0{width}}'.format(
                instance_id=instance_id_counter, width=max_string_padd)
            
            instance_id_counter += 1
            
            images, annotations = self.write_instance(
                instance, key
            )

            if df_images is None:
                df_images = images
            else:
                df_images = df_images.vstack(images)

            if df_annotations is None:
                df_annotations = annotations
            else:
                if annotations is not None:
                    df_annotations = df_annotations.vstack(annotations)

            if df_images.estimated_size() * 1e-9 > self.max_file_size:
                fname = 'images_{file_id:0{width}}.arrow'.format(
                    file_id=file_size_counter, width=max_string_padd)
                df_images.write_ipc(join(
                    self.__output__,
                    fname
                ))
                file_size_counter += 1
                df_images = None

        fname = 'boxes_{file_id:0{width}}.arrow'.format(
            file_id=1, width=max_string_padd)
        df_annotations.write_ipc(join(
            self.__output__,
            fname
        ))

        if df_images is not None:
            fname = 'images_{file_id:0{width}}.arrow'.format(
                file_id=file_size_counter, width=max_string_padd)
            df_images.write_ipc(join(
                self.__output__,
                fname
            ))

