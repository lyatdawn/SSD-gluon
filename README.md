# SSD-gulon
* gulon implementation of SSD in the paper "SSD: Single Shot MultiBox Detector"[[Paper]](https://arxiv.org/pdf/1512.02325.pdf).
* This code borrowed from https://zh.gluon.ai/chapter_computer-vision/ssd.html.

## Install Required Packages
First ensure that you have installed the following required packages:
* gulon. gulon is the interface of mxnet. The version of mxnet is mxnet0.12.0, maybe other version is ok.
* Opencv ([instructions](https://github.com/opencv/opencv)). Here is opencv-2.4.9.

## Datasets
* In the implementation of the SSD, we will use PASCAL VOC2007 and VOC2012 datasets. Run **scripts/make_DetIter/prepare_pascal.sh** to build Record data for detection. You may change im2rec.py a little, i.e. input str, then transform to int. We only modify resize, quality, num-thread, for example:
```
if type(args.resize) == str:
        args.resize = int(args.resize)
```

## Training and Validating a Model
* Run the following script to train and validate the model.
```shell
python trainval_ssd.py
```
You could change some arguments in the trainval_ssd.py, like num_epochs, gpus.

## Testing a Model
* Run the following script to test the model.
```shell
python predict.py
```