# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS

from glob import glob
import numpy as np
import polars as pl
from os.path import join, exists, splitext, basename
from typing import Union, Iterable
from PIL import ImageFile
from deepview.datasets.utils.progress import FillingSquaresBar

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
    ) -> None:
        super().__init__(
            classes=classes,
            silent=silent,
            shuffle=shuffle
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

        if isinstance(images, str):
            if "*" not in annotations and not exists(images):
                raise FileNotFoundError(
                    f"\n\t - [ERROR] Images folder does not exist at: {images}"
                )

            image_files = []
            for ext in ['*.[pP][nN][gG]',
                        '*.[jJ][pP][gG]',
                        '*.[jJ][pP][eE][gG]']:
                partial = glob(join(images, ext))
                image_files = image_files + partial
            self.images = image_files
        else:
            for image in images:
                if exists(image):
                    self.images.append(image)

                if not exists(image) and not self.silent:
                    print(
                        f"\t - [WARNING] File does not exist: {image}"
                    )

        if len(self.images) == 0:
            print(
                f"\n\t - [WARNING]  Aborting reading because no images were found in parameter ``{images}``"
            )
            exit(0)

        pbar = None
        if not self.silent:
            pbar = FillingSquaresBar(
                desc="\t [INFO] Reading: ",
                size=30,
                color="green",
                steps=len(self.images)
            )

        if isinstance(annotations, str):
            if "*" not in annotations and not exists(annotations):
                raise FileNotFoundError(
                    f"\n\t - [ERROR] Annotations folder does not exist at: {annotations}"
                )

            for image in self.images:
                image_name = splitext(basename(image))[0]
                
                if "*" in annotations:
                    ann_path = splitext(image)[0] + ".txt"
                    if exists(ann_path):
                        self.annotations.append(ann_path)
                        ann_file = ann_path
                    else:
                        ann_file = None
                else:
                    ann_file = join(annotations, image_name + '.txt')
                    if exists(ann_file):
                        self.annotations.append(ann_file)
                    else:
                        ann_file = None
                
                self.__storage__.append([image, ann_file])
                self.__size__ += 1
                
                if pbar:
                    pbar.update()
        else:
            pbar = FillingSquaresBar(
                desc="\t [INFO] Reading: ",
                size=30,
                color="green",
                steps=len(self.images)
            )

            for image, ann_file in zip(self.images, annotations):
                if not exists(ann_file):
                    ann_file = None
                else:
                    self.annotations.append(ann_file)

                self.__storage__.append([image, ann_file])
                self.__size__ += 1
                pbar.update()

        if len(self.annotations) == 0:
            print(
                f"\n\t - [WARNING]  Aborting reading because no annotation files were found in parameter ``{annotations}``"
            )
            exit(0)

        self.__current__ = 0

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
        shuffle: bool = False
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
            shuffle=shuffle
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
        shuffle: bool = False
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
            shuffle=shuffle
        )
