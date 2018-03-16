# -*- coding:utf-8 -*-
"""
gulon implement of SSD: Single Shot MultiBox Object Detector.
"""
import os
import argparse
import logging
import mxnet as mx
from mxnet import gluon
from net import SSD
from data import get_iterators
import utils

# logging
log_file = "./model/SSD_300x300.log" # SSD with 300 * 300 images.
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                   filename=log_file,
                   level=logging.INFO,
                   filemode='a+')
logging.getLogger().addHandler(logging.StreamHandler())

if __name__ == '__main__':
    # model, save checkpoint and training/valiing log.
    if not os.path.exists("model"):
        os.nakedirs("model")

    # parse args
    parser = argparse.ArgumentParser(description="train LightCNN-9 and LightCNN-29.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Not need data augmentations.
    parser.set_defaults(
        # data
        data_dir = "./datasets/VOC0712", # We transfer data_dir directly. PASCAL VOC.
        # data_dir = "./datasets/pikachu", # pikachu
        batch_size = 64,
        data_shape = (3, 300, 300), # 300 * 300.
        # rgb_mean = nd.array([123, 117, 104]), # not need currently.
        # class_names = [""], # Use class_names when predicting the box of giving image.
        # colors = [""], # Use colors when predicting the class of giving image.
        num_classes = 20, # In the succeeding process, num_classes will plus 1.
        # anchor box sizes and ratios for 5 feature scales. May have other choices.
        sizes = [[.2, .272], [.37, .447], [.54, .619], 
                      [.71, .79], [.88, .961]],
        ratios = [[1, 2, .5]] * 5, 
        # [[1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5]].

        # train and val
        disp_batches = 100,
        num_epochs = 1, # 30, epoch from 0 to 29.
        # optim
        # From random init to train.
        optimizer = 'adam', 
        lr = 0.001, # init lr is 1e-3.
        # lr_step_epochs, the epochs to reduce the lr, e.g. lr_step_epochs = '15,20'.
        lr_step_epochs = None,
        # lr_step_epochs = '26,28,30,32,34',
        # lr_decay, the ratio to reduce lr on each step. e.g. lr_decay = 0.1.
        lr_decay = 0.1,

        # chechpoint
        # load_epoch. Load trained model, load_epoch is the epoch of the model. e.g. load_epoch = 28.
        load_epoch = 0, # Load trained model. if load_epoch is 0, represent from random init to train.
        # model_prefix, the prefix of save checkp, e.g., SSD_300x300.params.
        model_prefix = 'model/SSD_300x300',
        )
    args = parser.parse_args()

    # context
    ctx = utils.try_gpu()

    # network
    net = SSD(num_classes=args.num_classes, sizes=args.sizes, ratios=args.ratios)
    # There are 21 classes, the first class is background. label form 0 to 20.
    # In the succeeding process, num_classes will plus 1.

    # init weight and bias
    net.initialize(ctx=ctx, init=mx.initializer.Xavier(magnitude=2))
    # initialize() define in mxnet/gulon/parameter.py.

    # Loss. Loss will defined in utils.py.

    # training and validating data. Use data.py to load data iter.
    train, val = get_iterators(args)

    # net parameters.
    net.collect_params().reset_ctx(ctx)
    # net.hybridize() # In SSD, only can use NDArray, Symbol is wrong.
    # Before net.hybridize(), F is using NDArray. atfer net.hybridize(), F is using Symbol.
    # Symbol code will not use python, but use C++ to compute!

    # Refer to http://zh.gluon.ai/chapter_computer-vision/kaggle-gluon-cifar10.html, set lr scheduler.
    # The trainer is defined in the main function. The lr scheduler is used in utils.py.
    '''
    In mxnet/gulon/trainer.py, the class Trainer has these method:
    1) learning_rate(), return current learning rate.
    2) set_learning_rate(lr), set the learning rate will be used.
    3) step(batch_size, ignore_stale_grad=False), Makes one step of parameter update based on batch_size data.
    4) save_states(fname), Saves trainer states (e.g. optimizer, momentum) to a file.
    5) load_states(fname), Loads trainer states (e.g. optimizer, momentum) from a file.

    So, we can use learning_rate() and set_learning_rate(lr) set lr scheduler.
    use trainer.learning_rate!
    '''
    trainer = gluon.Trainer(net.collect_params(),
              'sgd', {'learning_rate': 0.1, 'wd': 5e-4})
              # args.optimizer, {'learning_rate': args.lr})

    utils.train(batch_size=args.batch_size,
                train_data=train, 
                test_data=val, 
                net=net,
                trainer=trainer,
                ctx=ctx,
                num_epochs=args.num_epochs,
                lr_step_epochs=args.lr_step_epochs,
                print_batches=args.disp_batches,
                load_epoch=args.load_epoch,
                model_prefix=args.model_prefix)