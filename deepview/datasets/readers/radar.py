
from typing import Iterable, Union
from deepview.datasets.readers.darknet import DarknetDetectionReader
import numpy as np
import polars as pl
from PIL import Image, ImageFile
from os.path import exists

class DarknetDetectionRaivin2D(DarknetDetectionReader):
    def __init__(
        self,
        images: Union [str,  Iterable],
        annotations: Union [str, Iterable],
        classes: Union[str,  Iterable],
        bev: bool = True,
        silent: bool = False,
        out_format: str = "xywh",
        shuffle: bool = False,
        class_mask: set = None,
        from_radar: bool = True,
        fusion: bool = False,
        with_distances: bool = False
    ) -> None:
        super().__init__(
            images,
            annotations,
            classes,
            silent,
            out_format,
            shuffle,
            class_mask,
            look_for_files=["*.cube.npy"] if from_radar else ["*.jpeg"], # has to be a list or tuple or a set...
            annotations_as='.3dtxt'
        )
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.__bev__ = bev
        self.__load_annotations__ = self.load_bev if bev else self.load_2d_boxes_from_3d_boxes
        self.__load_input__ = self.load_cube if from_radar else self.load_rgb
        self.get_item = self.load_instance if not fusion else self.load_fusion_instance
        self.num_features = 6 if with_distances else 5
        self.use_distances = with_distances
        
        self.input_channels = 16
        self.range_bins = 192
        self.doppler_bins = 128
    
   
    def __getitem__(self, item) -> tuple:
        return self.get_item(item)
    
    def load_cube(self, file: str):
        cube = np.load(file).astype(np.float32)        
        return  cube
    
    def load_rgb(self, file):
        return np.asarray(
            Image.open(file).convert('RGB'),
            dtype=np.uint8
        )
    
    def load_instance(self, item):
        input_file, ann_file = self.__storage__[item]
        
        input_data = self.__load_input__(input_file)

        if ann_file is None:
            return input_data, np.array([], dtype=np.float32)

        try:
            boxes = pl.read_csv(
                ann_file,
                has_header=False,
                separator=" "
            )
            if self.__class_mask__ is not None:
                boxes = boxes.filter(
                    pl.col("column_1").is_in(self.__class_mask__))
            boxes = boxes.to_numpy()
        except pl.exceptions.NoDataError:
            return input_data, np.array([], dtype=np.float32)

        if len(boxes) == 0:
            return input_data, np.array([], dtype=np.float32)

        boxes = self.__load_annotations__(boxes)
        boxes = boxes.reshape(-1, self.num_features)
        boxes = boxes.astype(np.float32)

        return input_data, boxes
    
    def load_fusion_instance(self, item):
        input_file, ann_file = self.__storage__[item]        
        input_data = self.__load_input__(input_file)
        file = ann_file.replace(".3dtxt", ".jpeg")
        
        if exists(file):
            image = self.load_rgb(file)
        else:
            image = np.zeros(shape=(640, 640, 3), dtype=np.uint8)
            # print(f"missing: {file}")
        

        if ann_file is None:
            return image, input_data, np.array([], dtype=np.float32)

        try:
            boxes = pl.read_csv(
                ann_file,
                has_header=False,
                separator=" "
            )
            if self.__class_mask__ is not None:
                boxes = boxes.filter(
                    pl.col("column_1").is_in(self.__class_mask__))
            boxes = boxes.to_numpy()
        except pl.exceptions.NoDataError:
            return image, input_data, np.array([], dtype=np.float32)

        if len(boxes) == 0:
            return image, input_data, np.array([], dtype=np.float32)

        boxes = self.__load_annotations__(boxes)
        boxes = boxes.reshape(-1, self.num_features)
        boxes = boxes.astype(np.float32)

        
        return image, input_data, boxes

    def load_bev(self, boxes):
        # given: class, z, x, y, l, w, h -> [x, z, w, l, class]
        boxes = boxes.reshape(-1, 7)
        boxes = boxes[:, (0, 2, 1, 3, 5, 4, 6)]
        if self.use_distances:
            boxes = boxes[:, [1, 2, 4, 5, 0]]
        else:
            boxes = boxes[:, [1, 2, 4, 5, 1, 0]]
        return boxes
        

    def load_2d_boxes_from_3d_boxes(self, boxes):
        cam_mtx = np.array([
            [1260/1920, 0, 960/1920,],
            [0, 1260/1080, 540/1080,],
            [0, 0, 1,],
        ])

        # Converts lzxydwh (label, z, x, y, depth, width, height) coulmn vector into (x_min, y_min, z_min) column vector
        zxydwh2xyzmin = np.array([
            [0, 0, 1, 0, 0, -0.5, 0],
            [0, 0, 0, 1, 0, 0, -0.5],
            [0, 1, 0, 0, 0, 0, 0],
        ])

        # Converts lzxydwh (label, z, x, y, depth, width, height) coulmn vector into (x_max, y_max, z_max) column vector
        zxydwh2xyzmax = np.array([
            [0, 0, 1, 0, 0, 0.5, 0],
            [0, 0, 0, 1, 0, 0, 0.5],
            [0, 1, 0, 0, 0, 0, 0],
        ])

        # Converts (x, y, 1) column vector from camera coordinate system to the image coordinate system.
        coord_cnvt = np.array([
            [-1, 0, 1],
            [0, -1, 1],
            [0, 0, 1],
        ])
        xyzmin = zxydwh2xyzmin @ boxes.transpose()
        xyzmax = zxydwh2xyzmax @ boxes.transpose()
        bbmin = cam_mtx @ xyzmin
        bbmin /= bbmin[2, :]
        bbmin = coord_cnvt @ bbmin
        bbmin = bbmin[:2, :]

        bbmax = cam_mtx @ xyzmax
        bbmax /= bbmax[2, :]
        bbmax = coord_cnvt @ bbmax
        bbmax = bbmax[:2, :]

        bbmax_ = np.maximum(bbmax, bbmin)
        bbmin = np.minimum(bbmax, bbmin)
        sizes = bbmax_- bbmin
        mins2d, size2d = bbmin.transpose(), sizes.transpose()

        if self.use_distances:
            boxes = np.concatenate([
                mins2d + size2d * 0.5,
                size2d,
                boxes[:, 1:2],
                boxes[:, 0:1],
            ], axis=-1)
        else:
            boxes = np.concatenate([
                mins2d + size2d * 0.5,
                size2d,
                boxes[:, 0:1],
            ], axis=-1)

        return boxes

        
