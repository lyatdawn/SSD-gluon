# -*- coding:utf-8 -*-
"""
Utlize codes.
"""
import os
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.contrib.ndarray import MultiBoxTarget
# from mxnet.contrib.symbol import MultiBoxTarget
from mxnet import metric
from mxnet import autograd
from mxnet import nd
import mxnet as mx
import time
import logging

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list

# SSD -- anchor box, see scripts/generate_anchor_box.py for detial.

# Predict the class of object, see utils.py for detial.
'''
对每一个锚框我们需要预测它是不是包含了我们感兴趣的物体, 还是只是背景. 即预测每一个锚框覆盖的像素是不是包
含object. 这里我们使用一个3x3的卷积层做预测, 加上padding=1, 使用它的输出和输入一样. 
同时输出的通道数是num_anchors * (num_classes + 1)即卷积核的个数, 每个通道对应于一个锚框对某个类的置信度.
假设输出是Y, 那么对应输入中第n个样本的第(i,j)像素的置信值是在Y[n, :, i, j]里.
具体来说, 对于以(i, j)为中心的第a个锚框,
- 通道a * (num_class + 1), 是其只包含背景的score.
- 通道a * (num_class + 1) + 1 + b, 是其包含第b个物体的score.

我们定义个一个这样的类别分类器函数.
'''
def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(channels=num_anchors * (num_classes + 1), kernel_size=3, padding=1)
# Usage:
'''
cls_pred = class_predictor(5, 10)
cls_pred.initialize()
x = nd.zeros((2, 3, 20, 20))
y = cls_pred(x)
y.shape # [batch_size, num_channels, 20, 20].
'''

# Predict the bounding box, see utils.py for detial.
'''
因为真实的边界框可以是任意形状, 我们需要预测如何从一个锚框变换成真正的边界框. 
这个变换可以由一个长为4的向量来描述. 同上一样, 我们使用一个有num_anchors * 4通道的卷积.
假设输出是Y, 那么对应输入中第n个样本的第(i, j)像素为中心的锚框的转换在Y[n,:,i,j]里.
具体来说, 对于第a个锚框, 它的变换在a * 4到a * 4 + 3通道里.
'''
def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(channels=num_anchors * 4, kernel_size=3, padding=1)
# Usage
'''
box_pred = box_predictor(10)
box_pred.initialize()
x = nd.zeros((2, 3, 20, 20))
y = box_pred(x)
y.shape
'''

# Merge the predicted output from other layers.
'''
前面我们提到过SSD的一个重要性质是它会在多个层同时做预测. 每个层由于长宽和锚框选择不一样, 
导致输出的数据形状会不一样. 这里我们用物体类别预测作为样例, 边框预测是类似的.

为了之后处理简单, 我们将不同层的输入合并成一个输出. 
首先我们将通道移到最后的维度, 然后将其展成2D数组. 因为第一个维度是样本个数, 所以不同输出之间是不变.
我们可以将所有输出在第二个维度上拼接起来.
'''
# Merge the predicted output from other layers, see utils.py for detial.
# First we will flatten all the outputs to 2D array, then concat these at second dim, i.e. dim=1.
def flatten_prediction(pred):
    return pred.transpose(axes=(0, 2, 3, 1)).flatten()
    # NDArray call transpose() to permute the dimensions of an array.
    # call flatten() to flatten the input array into a 2-D array. 
    # The first dim of result is constant, i.e. batch_size. The second dim is channel * height * width. 

def concat_predictions(preds):
    return nd.concat(*preds, dim=1)
    # NDArray call concat() to concat the arrays through given axis/dim. The arguments are:
    '''
    data: List of arrays to concatenate.
    dim: the dimension to be concated.
    return NDArray.
    '''
# Usage
'''
flat_y1 = flatten_prediction(y1)
print('Flatten class prediction 1', flat_y1.shape)
flat_y2 = flatten_prediction(y2)
print('Flatten class prediction 2', flat_y2.shape)
y = concat_predictions([flat_y1, flat_y2])
print('Concat class predictions', y.shape)
'''

# Train.
'''
对于分类任务, 我们通过交叉熵来计算loss. 但物体检测里我们需要预测边框.
这里我们先引入一个概率来描述两个边框的距离-IoU. 交并比.

IoU: 交并比.
我们知道判断两个集合的相似度最常用的衡量叫做Jaccard距离, 给定集合A和B, Jaccard距离的定义是:

J(A, B) = \frac{|A \cap B|}{| A \cup B|} # \cap取交集; \cup取并集.

边框可以看成是像素的集合, 我们可以类似的定义它. 这个标准通常被称之为Intersection over Union (IoU).
IoU值越大表示两个边框很相似, 越小则两个边框不相似。

虽然每张图片里面通常只有几个标注的边框, 但SSD会生成大量的锚框. 可以想象很多锚框都不会框住感兴趣的物体, 
就是说锚框跟任何感兴趣物体的IoU都小于某个阈值. 这样就会产生大量的负类锚框, 或者说对应标号为0的锚框.
对于这类锚框有两点要考虑的:
1. 边框预测的损失函数不应该包括负类锚框, 因为它们并没有对应的真实边框.
2. 因为负类锚框数目可能远多于其他, 我们可以只保留其中的一些. 而且是保留那些目前预测最不确信它是负类的, 
   就是对类预测值排序, 选取数值最小的哪一些困难的负类锚框.
我们可以使用MultiBoxTarget来完成上面这两个操作.
'''
def training_targets(anchors, class_preds, labels):
    class_preds = class_preds.transpose(axes=(0, 2, 1))
    return MultiBoxTarget(anchors, labels, class_preds)
    '''
    MultiBoxTarget, Compute Multibox training targets. The arguments are:
    anchor: NDArray, Multibox prior anchor boxes.
    label: NDArray.
    overlap_threshold: default=0.5.
    '''
# Usage
# out = training_targets(anchors, class_preds, batch.label[0][0:1]) 
'''
training_targets()返回三个NDArray, 分别是:
1. 预测的边框跟真实边框的偏移, 大小是batch_size x (num_anchors * 4).
2. 用来遮掩不需要的负类锚框的掩码, 大小跟上面一致.
3. 锚框的真实的标号, 大小是batch_size x num_anchors.
'''

# Loss.
# Classification loss.
'''
对于分类问题, 最常用的损失函数是交叉熵. 这里我们定义一个类似于交叉熵的损失, 不同于交叉熵的定义:
log(p_j), 这里j是真实的类别, 且p_j是对于的预测概率. 我们使用一个称为关注损失的函数. 给定正的gamma和alpha, 
它的定义是: 
-alpha * (1 -p _j)^{gamma} * log(p_j)

演示不同gamma导致的变化. 可以看到, 增加gamma可以使得对正类预测值比较大时损失变小.
'''
# 这个自定义的损失函数可以简单通过继承gluon.loss.Loss来实现.
class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._batch_axis = batch_axis

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pj = output.pick(label, axis=self._axis, keepdims=True)
        loss = -self._alpha * ((1 - pj) ** self._gamma) * pj.log()
        return loss.mean(axis=self._batch_axis, exclude=True)
# Usage
# cls_loss = FocalLoss()

# Bounding box predict loss.
'''
对于边框的预测是一个回归问题. 通常可以选择平方损失函数(L2损失). 但这个损失对于比较大的误差的惩罚很高,
我们可以采用稍微缓和一点绝对损失函数(L1损失), 它是随着误差线性增长, 而不是平方增长. 
但这个函数在0点处导数不唯一, 因此可能会影响收敛. 一个通常的解决办法是在0点附近使用平方函数使得它更加平滑.
它被称之为平滑L1损失函数. 它通过一个参数sigma来控制平滑的区域:
       | (sigma x)^2/2, x < 1/sigma^2
f(x) = |
       |
       | |x|-0.5/sigma^2, otherwise

图示sigma的平滑L1损失和L2损失的区别. mxnet include Smooth L1 Loss,  mxnet.ndarray.smooth_l1.
'''
# 我们同样通过继承Loss来定义这个损失. 同时它接受一个额外参数mask, 这是用来屏蔽掉不需要被惩罚的负例样本.
class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)
        self._batch_axis = batch_axis

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return loss.mean(self._batch_axis, exclude=True)
# box_loss = SmoothL1Loss()

def evaluate_accuracy(data_iterator, net, ctx, cls_metric, box_metric):
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()

    for _, batch in enumerate(data_iterator):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
        
        anchors, class_preds, box_preds = net(x)
        box_target, box_mask, cls_target = training_targets(
            anchors, class_preds, y)
        
        # update metrics
        cls_metric.update([cls_target], [class_preds.transpose((0, 2, 1))])
        box_metric.update([box_target], [box_preds * box_mask])

    return cls_metric, box_metric

def train(batch_size, train_data, test_data, net, trainer, ctx, num_epochs, 
        lr_step_epochs=None, lr_decay=0.1, print_batches=100, load_epoch=0, model_prefix=None, 
        period=1):
    """
    Train a network.
    required=True for those uninitialized arguments.
    Refer to mxnet/module/base_module.py fit() to load trained model.
    Refer to fit.py to set lr scheduler.
    """
    logging.info("Start training on {}".format(ctx))
    # Load trained model.
    # Indicates the starting epoch. Usually, if resumed from a checkpoint saved at a previous training phase 
    # at epoch N, then this value is N.
    if load_epoch > 0:
        if os.path.exists(model_prefix + "-{}.params".format(load_epoch)):
            net.load_params(model_prefix + "-{}.params".format(load_epoch), ctx)
            logging.info("Resume training from epoch {}".format(load_epoch))
        else:
            print("The resume model does not exist.")

    # if isinstance(ctx, mx.Context):
    #     ctx = [ctx]
    # Not use.

    # Set lr scheduler.
    # can use learning_rate() and set_learning_rate(lr) to set lr scheduler.
    if lr_step_epochs is not None:
        step_epochs = [int(l) for l in lr_step_epochs.split(',')]
        for s in step_epochs:
            if epoch == s:
                trainer.set_learning_rate(trainer.learning_rate * lr_decay)
                logging.info("Adjust learning rate to {} for epoch {}".format(trainer.learning_rate, epoch))
                # Use trainer.learning_rate

    # Loss
    cls_loss = FocalLoss()
    box_loss = SmoothL1Loss()

    # Evaluate.
    '''
    对于分类好坏我们可以沿用之前的分类精度. 
    评估边框预测的好坏的一个常用是是平均绝对误差. 但是平方误差对于大的误差给予过大的值, 从而数值上过于敏感.
    平均绝对误差就是将二次项替换成绝对值, 具体来说就是预测的边框和真实边框在4个维度上的差值的绝对值.
    '''
    cls_metric = metric.Accuracy() # classification evaluation.
    box_metric = metric.MAE() # box prediction evaluation.
    # validating
    val_cls_metric = metric.Accuracy() # classification evaluation.
    val_box_metric = metric.MAE() # box prediction evaluation.

    # the CUDA implementation requres each image has at least 3 lables. 
    # Padd two -1 labels for each instance. Use when loading pikachu dataset. ???
    # train_data.reshape(label_shape=(3, 5))
    # train_data = train_data.sync_label_shape(train_data)

    for epoch in range(load_epoch, num_epochs):
        train_loss, n = 0.0, 0.0
        # reset data iterators and metrics. Must reset!
        train_data.reset()
        cls_metric.reset()
        box_metric.reset()
        val_cls_metric.reset()
        val_box_metric.reset()

        tic = time.time()
        for i, batch in enumerate(train_data):
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)

            with autograd.record():
                anchors, class_preds, box_preds = net(x)
                box_target, box_mask, cls_target = training_targets(
                    anchors, class_preds, y)
                # losses
                loss1 = cls_loss(class_preds, cls_target)
                loss2 = box_loss(box_preds, box_target, box_mask)
                loss = loss1 + loss2
            loss.backward()
            train_loss += sum([l.sum().asscalar() for l in loss])
            trainer.step(batch_size)
            n += batch_size
            # update metrics
            cls_metric.update([cls_target], [class_preds.transpose((0, 2, 1))])
            box_metric.update([box_target], [box_preds * box_mask])

            if print_batches and (i+1) % print_batches == 0:
                logging.info(
                    "Epoch [%d]. Batch [%d]. Loss [%f]. Time %.1f sec" % 
                    (epoch, n, train_loss/n, time.time() - tic))
                # cls_metric.get() will return a NDArray (string, float).
                # print
                print("Train acc:", cls_metric.get(), box_metric.get())

        val_cls_metric, val_box_metric = evaluate_accuracy(test_data, net, ctx, val_cls_metric, 
            val_box_metric)
        # print
        print("Val acc: ", val_cls_metric.get(), val_box_metric.get())

        # save checkpoint
        if (epoch + 1) % period == 0:
            net.save_params(model_prefix + "-{}.params".format(epoch + 1))
            logging.info("Saved checkpoint to {}-{}.params".format(model_prefix, epoch + 1))