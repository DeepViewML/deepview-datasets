
from typing import Iterable
from deepview.datasets.readers.core import DarknetDetectionReader


class DarknetDetectionRaivin2D(DarknetDetectionReader):
    def __init__(
        self,
        images: str | Iterable,
        annotations: str | Iterable,
        classes: str | Iterable,
        bev: bool = True,
        silent: bool = False,
        out_format: str = "xywh",
        shuffle: bool = False,
        class_mask: set = None
    ) -> None:
        super().__init__(
            images,
            annotations,
            classes,
            silent,
            out_format,
            shuffle,
            class_mask,
            look_for_files="*.cube.npy",
            annotations_as='.3dtxt'
        )
        self.__bev__ = bev
        self.__load_input_data__ = None
        self.__load_annotations__ = None

    def __getitem__(self, item) -> tuple:
        return super().__getitem__(item)


class DarknetDetectionRaivin3D(DarknetDetectionReader):
    raise NotImplementedError("Not Implemented Class")
