# SSD-gulon
* gulon implementation of SSD in the paper "SSD: Single Shot MultiBox Detector"[[Paper]](https://arxiv.org/pdf/1512.02325.pdf).
* This code borrowed from https://zh.gluon.ai/chapter_computer-vision/ssd.html.

## Install Required Packages
First ensure that you have installed the following required packages:
* gulon. gulon is the interface of mxnet. The version of mxnet is mxnet0.12.0, maybe other version is ok.
* Opencv ([instructions](https://github.com/opencv/opencv)). Here is opencv-2.4.9.

## Training a Model
* Run the following script to train and validate the model.
```shell
python trainval_ssd.py
```
You could change some arguments in the trainval_ssd.py, like num_epochs, gpus.