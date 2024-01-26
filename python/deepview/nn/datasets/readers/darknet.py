# Copyright 2022 by Au-Zone Technologies.  All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential.
#
# This source code is provided solely for runtime interpretation by Python.
# Modifying or copying any source code is explicitly forbidden.

from deepview.nn.datasets.readers.core import BaseReader
from os.path import join, exists, splitext, basename
from typing import Union, Iterable
from PIL import Image, ImageFile
from glob import glob
from tqdm import tqdm
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True

class DarknetReader(BaseReader):
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
        silent: bool = False
    ) -> None:
        super().__init__(
            classes=classes,
            silent=silent
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
        
        self.images = []
        self.annotations = []
        self.__size__ = 0
        self.__current__ = -1
        
        if isinstance(images, str):
            if not exists(images):
                raise FileNotFoundError(
                    f"\n\t - [ERROR] Images folder does not exist at: {images}"
                )
                
            image_files = []
            for ext in ['*.[pP][nN][gG]', '*.[jJ][pP][gG]', '*.[jJ][pP][eE][gG]']:
                partial = glob(join(images, ext))
                image_files = image_files + partial
            self.images = image_files
        else:
            for image in images:
                if not exists(image) and not self.silent:
                    print(
                        f"\t - [WARNING] File does not exist: {image}"
                    )
                else:
                    self.images.append(image)
            
        
        if len(self.images) == 0:
            print(
                f"\n\t - [WARNING]  Aborting reading because no images were found in parameter ``images``"
            )
            exit(0)
        
        
        if self.silent:
            loop = self.images
        else:
            loop = tqdm(
                self.images, 
                desc="\t [INFO] Reading",
                colour="green",
                bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}'
            )
        
        if isinstance(annotations, str):
            if not exists(annotations):
                raise FileNotFoundError(
                    f"\n\t - [ERROR] Images folder does not exist at: {annotations}"
                )
            
            for image in loop:
                image_name = splitext(basename(image))[0]
                ann_file = join(annotations, image_name + '.txt')
                
                
                if not exists(ann_file):
                    ann_file = None
                else:
                    self.annotations.append(ann_file)
                    
                self.__storage__.append([image, ann_file])
                self.__size__ += 1
                
        else:
            for ann_file in loop:                
                if not exists(ann_file):
                    ann_file = None
                else:
                    self.annotations.append(ann_file)
                self.__storage__.append([image, ann_file])
                self.__size__ += 1
        
        if len(self.annotations) == 0:
            print(
                f"\n\t - [WARNING]  Aborting reading because no annotation files were found in parameter ``annotations``"
            )
            exit(0)    
        
        self.__current__ = 0
        
    
    def __getitem__(
        self, 
        item: int
    ) -> tuple:
        """
        This function reads the content from the image file and returns a tuple with the following structure:
        
        Example:

            ::
                reader = DarknetReader(...)
                image, annotation_path   = reader[0]
                
                # image: np.ndarray with internal type np.uint8 (Image content)
                # annotation_path: Path to the annotation file. This is handled by specific child Reader classes
                # ``DarknetDetectionReader`` and ``DarknetSegmentationReader``


        Parameters
        ----------
        item : int
            Index of the element to be retrieved

        Returns
        -------
        tuple
            (np.ndarray(np.uint8), str): image content and path to annotation file
        """
        instance = super().__getitem__(item)
        image = Image.open(instance[0]).convert('RGB')
        image = np.asarray(image, dtype=np.uint8)
        return image, instance[1]
        

class DarknetDetectionReader(DarknetReader):
    
    def __init__(
        self, 
        images: Union[str, Iterable],
        annotations: [str, Iterable], 
        classes: Union[str, Iterable],
        silent: bool = False,
        out_format: str = "xywh"
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
            Either of a list containing the name of the classes or the path to a file containing the classes
            
        silent : bool, optional, default False
             Whether printing to the console or not, by default False
        
        out_format : str, default "xywh"
            This parameter specify the coordinate format for returning boxes
        
        
        Raises
        ------
        FileNotFoundError
            An exception is thrown in case path to images or annotations does not exist
        
        Return
        ------
        None
        
        """
        
        super().__init__(
            images=images, 
            annotations=annotations, 
            classes=classes,
            silent=silent
        )
        
        if not out_format in ["xywh", "xyxy"]:
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
        
        This function calls base class to read image from file and read annotations from file.
        The function will returns the image content as a np.ndarray(np.uint8) and a np.ndarray(np.float32) for boxes.
        
        Image is RGB and boxes will be a multidimensional array with shape  (N, 5).
        Boxes will be internally ordered as specified in ``out_format`` constructor parameter plus the class index at the end

        Parameters
        ----------
        item : int
            Id of the instance to be retrieved from ``storage``

        Returns
        -------
        tuple
            A tuple containing the real values of a single instance for object detection. The image and bounding boxes.
        """
        image, ann_file = super().__getitem__(item)
        
        if ann_file is None:
            return image, np.asarray([], dtype=np.float32)
        
        boxes = np.genfromtxt(ann_file)
        if len(boxes) == 0:
            return image, np.asarray([], dtype=np.float32)
        
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
        

class DarknetSegmentationReader(DarknetReader):
    def __getitem__(
        self, 
        item
    ) -> tuple:
        pass



if __name__ == '__main__':
    reader = DarknetDetectionReader(
        images="/home/reinier/development/modelpack-radar/datasets/modelpack/playing-cards-files/images/train",
        annotations="/home/reinier/development/modelpack-radar/datasets/modelpack/playing-cards-files/labels/train",
        classes=["ace", "nine", "six", "four", "eight", "queen", "seven", "king", "ten","jack", "five", "two","three"]
    )
    
    for image, boxes in reader:
        print(image.shape, boxes.shape)