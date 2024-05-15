# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from glob import glob
import numpy as np
import polars as pl
from os.path import join, exists, splitext, basename, dirname
from typing import Union, Iterable
from PIL import ImageFile
from deepview.datasets.utils.progress import FillingSquaresBar
import os
from deepview.datasets.readers.core import ObjectDetectionBaseReader

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DarknetReader(ObjectDetectionBaseReader):
    """
    Enables the abstract interface for reading darknet format.
    This format represents images and annotations as separated files.
    Each image file has an associated .txt file with the respective
    annotations.
    """

    def __init__(
        self,
        images: Union[str, Iterable],
        annotations: Union[str, Iterable],
        classes: Union[str, Iterable],
        silent: bool = False,
        shuffle: bool = False,
        groups: Iterable = None
    ) -> None:
        super().__init__(
            classes=classes,
            silent=silent,
            shuffle=shuffle,
            groups=groups
        )
        """
        Class constructor

        Parameters
        -----------
        images : Union[str, Iterable]

        annotations : Union[str, Iterable]
            Either of the path to the folder containng *.txt files or a list
            containing the path to all the

        classes:  Union[str, Iterable]
            Either of a list containing the name of the classes or the path to
            a file containing the classes

        silent : bool, optional, default False
             Whether printing to the console or not, by default False
        
        shuffle : bool, optional
            This parameter force data to be shuffled everytime the iterator ends, Default to False
        
        groups : Iterable, optional
            This parameter is used to identify candidate subfolders when are present. E.g. 
            /data/images/
                - day-1
                - day-2
            groups=[day1, day2] or groups=[day-1] or groups=None to include all of them
            This parameter is mostly used when reading raivin dataset. 
            As additional feature, the files could be grouped by folder or by filename
            
        Raises
        ------
        FileNotFoundError
            An exception is thrown in case path to images or annotations does
            not exist

        Return
        ------
        None

        """

        self.images = []
        self.annotations = []
        self.__size__ = 0
        self.__current__ = -1
        
        if not exists(images):
            raise FileNotFoundError(
                f"\n\t - [ERROR] Images folder does not exist at: {images}"
            )
            
        all_images = []
        for root, subdirs, files in os.walk(images):
            if len(subdirs) > 0:
                filtered_groups = self.groups if self.groups is not None else subdirs
                for group in filtered_groups: # reading groups as folders
                    for ext in ['*.[pP][nN][gG]',
                                '*.[jJ][pP][gG]',
                                '*.[jJ][pP][eE][gG]']:
                        all_images = all_images + glob(join(images, group, ext))
            else:
                if self.groups is None: # no groups
                    for ext in ['*.[pP][nN][gG]',
                            '*.[jJ][pP][gG]',
                            '*.[jJ][pP][eE][gG]']:
                        all_images = all_images + glob(join(images, ext))
                else:
                    for group in self.groups: # reading all the groups in the same folder
                        for ext in [f'{group}*.[pP][nN][gG]',
                            f'{group}*.[jJ][pP][gG]',
                            f'{group}*.[jJ][pP][eE][gG]']:
                            all_images = all_images + glob(join(images, ext))
            break # read a single level
        
        if not exists(images):
            raise FileNotFoundError(
                    f"\n\t - [ERROR] Annotations folder does not exist at: {annotations}"
            )
        
        pbar = FillingSquaresBar(desc="- Loading: ", size=30, steps=len(all_images), color='green')
        for image in all_images:
            ann_file = splitext(basename(image))[0] + '.txt'
            ann_path = join(annotations, ann_file)
            self.images.append(image)
            
            if exists(ann_path): # check the direct annotation folder
                self.__storage__.append([image, ann_path])
                self.annotations.append(ann_path)
                continue
            
            group_name  = basename(dirname(image))
            ann_path = join(annotations, group_name, ann_file)
            if exists(ann_path):
                self.__storage__.append([image, ann_path])
                self.annotations.append(ann_path)
            else:
                self.__storage__.append([image, None])
            
            pbar.update()
            
        self.__current__ = 0
        self.__size__ = len(self.__storage__)
        
    
    def __getitem__(
        self,
        item: int
    ) -> tuple:
        """
        This function reads the content from the image file and returns a tuple
        with the following structure:

        Example:

            ::
                reader = DarknetReader(...)
                image, annotation_path   = reader[0]

                # image: np.ndarray with internal type np.uint8 (Image content)
                # annotation_path: Path to the annotation file. This is handled
                # by specific child Reader classes
                # ``DarknetDetectionReader`` and ``DarknetSegmentationReader``


        Parameters
        ----------
        item : int
            Index of the element to be retrieved

        Returns
        -------
        tuple
            (np.ndarray(np.uint8), str): image content and path to annotation
            file
        """

        instance = super().__getitem__(item)
        self.__instance_id__ = splitext(basename(instance[0]))[0]

        image = np.fromfile(instance[0], dtype=np.uint8)
        return image, instance[1]

    def get_class_distribution(self) -> dict:
        pbar = FillingSquaresBar(
                desc="\t Loading classes: ",
                size=30,
                color="green",
                steps=len(self.annotations)
            )
        
        classes = []
        for ann in self.annotations:
            data = np.genfromtxt(ann)
            if len(data) == 0:
                continue
            if len(data.shape) == 1:
                data = np.expand_dims(data, 0)

            classes.append(data[:, 0])
            pbar.update()
        classes = np.concatenate(classes, axis=0).astype(np.int32)
        classes = np.bincount(classes)        
        return dict(enumerate(classes))
    

class DarknetDetectionReader(DarknetReader):

    def __init__(
        self,
        images: Union[str, Iterable],
        annotations: Union[str, Iterable],
        classes: Union[str, Iterable],
        silent: bool = False,
        out_format: str = "xywh",
        shuffle: bool = False,
        groups: Iterable = None
    ) -> None:
        """
        Class constructor

        Parameters
        -----------
        images : Union[str, Iterable]

        annotations : Union[str, Iterable]
            Either of the path to the folder containng *.txt files or a list
            containing the path to all the

        classes:  Union[str, Iterable]
            Either of a list containing the name of the classes or the path to
            a file containing the classes

        silent : bool, optional, default False
             Whether printing to the console or not, by default False

        out_format : str, default "xywh"
            This parameter specify the coordinate format for returning boxes

        shuffle : bool, optional
            This parameter force data to be shuffled everytime the iterator ends, Default to False

        Raises
        ------
        FileNotFoundError
            An exception is thrown in case path to images or annotations does
            not exist

        Return
        ------
        None

        """

        super().__init__(
            images=images,
            annotations=annotations,
            classes=classes,
            silent=silent,
            shuffle=shuffle,
            groups=groups
        )

        if out_format not in ["xywh", "xyxy"]:
            raise ValueError(
                f"Invalid output format for bounding boxes was provided: {out_format}"
            )

        self.box_format = out_format

    def to_xyxy(self, boxes: np.ndarray):
        """
        Transform boxes from xywh format into xyxy format

        Parameters
        ----------
        boxes : np.ndarray
            Multidimensional input array of shape (N, c) with c > 4

        Returns
        -------
        np.ndarray
            Transformed boxes into xyxy format
        """

        boxes = np.concatenate([
            boxes[:, [0, 1]] - boxes[: [2, 2]] * 0.5,
            boxes[:, [0, 1]] + boxes[: [2, 2]] * 0.5,
        ], axis=1)

        return boxes

    def __getitem__(
        self,
        item
    ) -> tuple:
        """

        This function calls base class to read image from file and read
        annotations from file. The function will returns the image content as a
        np.ndarray(np.uint8) and a np.ndarray(np.float32) for boxes.

        Image is RGB and boxes will be a multidimensional array with shape
        (N, 5). Boxes will be internally ordered as specified in ``out_format``
        constructor parameter plus the class index at the end

        Parameters
        ----------
        item : int
            Id of the instance to be retrieved from ``storage``

        Returns
        -------
        tuple
            A tuple containing the real values of a single instance for object
            detection. The image and bounding boxes.
        """
        data, ann_file = super().__getitem__(item)
        image = np.asarray(data, dtype=np.uint8)

        if ann_file is None:
            return image, np.array([], dtype=np.float32)

        try:
            boxes = pl.read_csv(ann_file, has_header=False,
                                separator=" ").to_numpy()
        except pl.exceptions.NoDataError:
            return image, np.array([], dtype=np.float32)

        if len(boxes) == 0:
            return image, np.array([], dtype=np.float32)

        if len(boxes.shape) == 1:
            boxes = boxes[None, :]

        boxes = boxes[:, [1, 2, 3, 4, 0]].astype(np.float32)

        if self.box_format == 'xywh':
            return image, boxes

        if self.box_format == 'xyxy':
            boxes = self.to_xyxy(boxes)
            return image, boxes

        raise RuntimeError(
            f"Something when wrong with annotation file: {ann_file}"
        )

    def get_boxes_dimensions(self) -> np.ndarray:
        
        pbar = FillingSquaresBar(
                desc="\t Loading boxes: ",
                size=30,
                color="green",
                steps=len(self.annotations)
            )
        bboxes = []
        for ann in self.annotations:
            data = np.genfromtxt(ann)
            if len(data) == 0:
                continue
            if len(data.shape) == 1:
                data = np.expand_dims(data, 0)
            bboxes.append(data[:, [3, 4]])
            pbar.update()
        return np.concatenate(bboxes, axis=0)


class UltralyticsDetectionReader(DarknetDetectionReader):
    def __init__(
        self,
        images: str,
        classes: Union[str, Iterable],
        silent: bool = False,
        out_format: str = "xywh",
        path: str = None,
        shuffle: bool = False,
        groups: Iterable = None
    ) -> None:
        """
        Class constructor

        Parameters
        -----------
        images : Union[str, Iterable]
            Path to a txt file that internally stores the path to each Image. In case internal path are relative 
            to a folder, the parent folder must be provided, otherwise it will assume current directory as base path.
            Notice that annotations files will be taken in the ultralitics format (https://github.com/ultralytics/ultralytics/issues/3087)

        classes:  Union[str, Iterable]
            Either of a list containing the name of the classes or the path to
            a file containing the classes

        silent : bool, optional, default False
             Whether printing to the console or not, by default False

        out_format : str, default "xywh"
            This parameter specify the coordinate format for returning boxes

        path : str, default None
            This parameter specifies the base path of the dataset. The value should be added to each 
            annotation within the txt file in order to get the relative path of each image and annotation. (YoloV5)
            In case this parameter is None, each line within the txt files will be taken as the relative path


        Raises
        ------
        FileNotFoundError
            An exception is thrown in case path to images or annotations does
            not exist

        Return
        ------
        None

        """
        if path and not exists(path):
            raise FileNotFoundError(
                F"Dataset base directory was not found at: {path}"
            )

        if not exists(images):
            images = join(path, images)
        if not exists(images):
            raise FileExistsError(
                f"Images file was not found at: {images}. It is not a relative path starting at: {path}"
            )

        with open(images, 'r') as fp:
            image_files = [line.rstrip() for line in fp.readlines()]

        if path:
            image_files = [join(path, image) for image in image_files]

        annotation_files = []
        for ann in image_files:
            ann = splitext(ann)[0] + '.txt'
            ann = ann.replace("images", "labels")
            annotation_files.append(ann)

        super(UltralyticsDetectionReader, self).__init__(
            images=image_files,
            annotations=annotation_files,
            classes=classes,
            silent=silent,
            out_format=out_format,
            shuffle=shuffle,
            groups=groups
        )
