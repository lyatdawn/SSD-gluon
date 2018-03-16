# -*- coding:utf-8 -*-
from __future__ import print_function
import sys, os
import argparse
import subprocess
import mxnet
from pascal_voc import PascalVoc
from mscoco import Coco
from concat_db import ConcatDB

def load_pascal(image_set, year, devkit_path, shuffle=False, class_names=None, true_negative=None):
    """
    wrapper function for loading pascal voc dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    year : str
        2007, 2012 or combinations splitted by comma
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """
    # image_set is the set, like trainval, val, test.
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"
    # year, only used while the dataset us PASCAL VOC.
    year = [y.strip() for y in year.split(',')]
    assert year, "No year specified"

    # make sure (# sets == # years)
    # list * integer, return a list. This operation will copy the elements of list for integer times.
    # Must ensure len(image_set) == len(year).
    if len(image_set) > 1 and len(year) == 1:
        year = year * len(image_set)
    if len(image_set) == 1 and len(year) > 1:
        image_set = image_set * len(year)
    assert len(image_set) == len(year), "Number of sets and year mismatch"

    imdbs = []
    for s, y in zip(image_set, year):
        '''
        For example:
        i=0, s="trainval", year="2007"
        i=1, s="trainval", year="2012"
        '''
        imdbs.append(PascalVoc(s, y, devkit_path, shuffle, is_train=True, class_names=class_names, 
            names="label_map.txt", true_negative_images=true_negative))
        '''
        devkit_path is the root path, can be "/home/ly/caffe-ssd/data/VOC0712".
        class_names, default is None.
        default class_names is None, so there needs a class file, which is the label_map.txt
            in caffe-ssd. There are 21 classes, the first class is background. label form 0 to 20.
            Must appoint the dirname.
        '''
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
        # class ConcatDB, ConcatDB is used to concatenate multiple imdbs to form a larger db.
    else:
        return imdbs[0]

def load_coco(image_set, dirname, shuffle=False):
    """
    wrapper function for loading ms coco dataset

    Parameters:
    ----------
    image_set : str
        train2014, val2014, valminusminival2014, minival2014
    dirname: str
        root dir for coco
    shuffle: boolean
        initial shuffle
    """
    anno_files = ['instances_' + y.strip() + '.json' for y in image_set.split(',')]
    assert anno_files, "No image set specified"
    imdbs = []
    for af in anno_files:
        af_path = os.path.join(dirname, 'annotations', af)
        imdbs.append(Coco(af_path, dirname, shuffle=shuffle))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='pascal', type=str) # which dataset will be uesd, like pascal, coco.
    parser.add_argument('--year', dest='year', help='which year to use',
                        default='2007,2012', type=str) # For PASCAL dataset, use VOC07 and VOC12 dataset.
    parser.add_argument('--set', dest='set', help='train, val, trainval, test',
                        default='trainval', type=str) # target dataset.
    parser.add_argument('--target', dest='target', help='output list file',
                        default=None,
                        type=str) 
    # The path and name of lst file, must specify. In the process of generating rec dataiter, it will
    # appoint target, e.g. ../../datasets/VOC0712/trainval.lst. 
    # So, os.path.abspath(args.target) is /home/ly/mxnet/example/ly/SSD-gulon/datasets/VOC0712/trainval.lst.
    # os.path.dirname(args.target) is /home/ly/mxnet/example/ly/SSD-gulon/datasets/VOC0712.
    # Use args.target appoint the path and name of saved lst file, then use args.target appoint the lst
    # file parent path. Finally, load lst file, it will only load *.lst file.
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default=None, help='string of comma separated names, or text filename')
    # 
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=None,
                        type=str)
    # dataset root path. must specify, like /home/ly/caffe-ssd/data/VOC0712.
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list',
                        type=bool, default=True)
    # shuffle.
    parser.add_argument('--true-negative', dest='true_negative', help='use images with no GT as true_negative',
                        type=bool, default=False)
    # Address no ground truth images.
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.class_names is not None:
        assert args.target is not None, 'for a subset of classes, specify a target path. Its for your own safety'
    if args.dataset == 'pascal':
        db = load_pascal(args.set, args.year, args.root_path, args.shuffle, args.class_names, 
            args.true_negative)
        # db is a object of class PascalVoc, and is also a object of class Imdb.
        # class Imdb, Base class for dataset loading.
        '''
        set: set can be "trainval", "trainval, train, test" and so on.
        year: it only use while the dataset is PASCAL VOC dataset. can be "2007", "2007, 2012".
        '''
        print("saving list to disk...")
        db.save_imglist(args.target, root=args.root_path)
        # class Imdb's object call save_imglist() to generate the lst file.
        # After getting the lst file, we can generate the rec dataset.
    elif args.dataset == 'coco':
        db = load_coco(args.set, args.root_path, args.shuffle)
        print("saving list to disk...")
        db.save_imglist(args.target, root=args.root_path)
    else:
        raise NotImplementedError("No implementation for dataset: " + args.dataset)

    print("List file {} generated...".format(args.target))
    # target can be train and val. After getting the lst file, we can generate the rec dataset.


    im2rec_path = os.path.join(mxnet.__path__[0], 'tools/im2rec.py')
    # final validation - sometimes __path__ (or __file__) gives 'mxnet/python/mxnet' instead of 'mxnet'
    if not os.path.exists(im2rec_path):
        im2rec_path = os.path.join(os.path.dirname(os.path.dirname(mxnet.__path__[0])), 'tools/im2rec.py')
    subprocess.check_call(["python", im2rec_path,
        os.path.abspath(args.target), os.path.abspath(args.root_path),
        "--shuffle", str(int(args.shuffle)), "--pack-label", "1", 
        "--resize", "300",  "--quality", "95", "--num-thread", "8"])
    '''
    python path/im2rec.py [flags]
    The arguments of im2rec.py.
    prefix: the first argument is the prefix of rec file, like train, val.
    root: the second argument is the root path of images, like /home/ly/caffe-ssd/data/VOC0712.
        The last column of lst file is the absolute path of image, e.g. VOC2007/JPEGImages/*.jpg.
        So, the second argument + the last column of lst file is the whole path of images.
    shuffle: train: True. val: False.
    --pack-label: True, Whether to also pack multi dimensional label in the record file. 
        In object detection task, pack-label must is True, since the label_width is changeable.
    --resize: 300. SSD_300X300
    --quality: 95. JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9.
    --color: default is 1, RGB images.
    --num-thread: 8.

    subprocess.check_call() must contain only strings, so we can change im2rec.py a little.
    input str, then transform to int. e.g.
    if type(args.resize) == str:
        args.resize = int(args.resize)
    
    Here, it will load the generated lst file automaticlly.
    '''
    
    print("Record file {} generated...".format(args.target.split('.')[0] + '.rec'))
