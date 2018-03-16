#!/usr/bin/env bash

# the type of assigned value is string.
# generate trainval.lst. When set is trainval, the number of images is 1/2.
python prepare_dataset.py \
       --dataset pascal \
       --year 2007,2012 \
       --set trainval \
       --target ../../datasets/VOC0712/trainval.lst \
       --root /home/ly/caffe-ssd/data/VOC0712

# generate val.lst
python prepare_dataset.py \
       --dataset pascal \
       --year 2007 \
       --set val \
       --target ../../datasets/VOC0712/val.lst \
       --root /home/ly/caffe-ssd/data/VOC0712 \
       --shuffle False
