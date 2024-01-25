# DeepView Dataset Library

This project contains python and rust binding for reading and exporting datasets. 
The library also includes Dataset iterators for both GPU (Tensorflow/Pytorch) and 
iterators for running validation on embedded devices where ML framewors are not 
available.

The labrary is distributed under `deepview.nn.datasets` and several ML tasks are included.

- Object Detection
- Distance Estimation
- Instance Segmentation and Semantic Segmentation
- Radar and Camera Fusion
- PCL classification

Data augmentation is another key feature included into this library. Augmenting data
is higly recommended when training ML models. The larger the model, the more data needs.





