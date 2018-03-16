# -*- coding:utf-8 -*-
"""
The network of SSD. SSD includes 4 modules: 
body network for extracting features.
3 down_sample for generating multi scale features.
5 classification modules.
5 box predicting modules.

Utlize gulon redefine the network. 
use net.hybridize()! Base class is nn.HybridBlock, and use nn.HybridSequential() construct a network.
In nn.HybridBlock, we should implement the hybrid_forward() method!
Symbol use the function, ndarray must can use! The name of function must be same!
"""
from mxnet.gluon import nn
from mxnet import sym
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior
# from mxnet.contrib.symbol import MultiBoxPrior
# import utlized functions.
from utils import class_predictor, box_predictor, flatten_prediction, concat_predictions

# Main network for extracting features. Could be ResNet.
class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                              strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                  strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)
'''
def Residual(channels, same_shape=True):
    strides = 1 if same_shape else 2
    
    out = nn.Sequential()
    out.add(
        nn.Conv2D(channels=channels, kernel_size=3, padding=1, strides=strides),
        nn.BatchNorm(),
        nn.Conv2D(channels=channels, kernel_size=3, padding=1), # default stride=1.
        nn.BatchNorm()
        )
    if not same_shape:
        out.add(nn.Conv2D(channels=channels, kernel_size=1, strides=strides))
    
    return out
'''
# Have problem, loss is nan! This problem might come from the dim.
def ResNet():
    # block 1
    b1 = nn.Conv2D(channels=64, kernel_size=7, strides=2)
    # block 2
    b2 = nn.Sequential()
    b2.add(
        nn.MaxPool2D(pool_size=2, strides=2),
        Residual(64),
        Residual(64)
        )
    # block 3
    b3 = nn.Sequential()
    b3.add(
        Residual(128, same_shape=False),
        Residual(128)
        )
    # block 4
    b4 = nn.Sequential()
    b4.add(
        Residual(256, same_shape=False),
        Residual(256)
        )
    # block 5
    b5 = nn.Sequential()
    b5.add(
        Residual(512, same_shape=False),
        Residual(512)
        )
    # chain all blocks together
    out = nn.Sequential()
    out.add(b1, b2, b3, b4)

    return out

# Down sample.
'''
Define a conv block for generating multi scale features, including two Conv-BatchNorm-Relu 
blocks and then a pooling layer to halve the feature size.
'''
def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer to halve the feature size"""
    out = nn.Sequential()
    for _ in range(2):
        out.add(nn.Conv2D(channels=num_filters, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(),
            nn.Activation("relu")
            )
    out.add(nn.MaxPool2D(pool_size=2, strides=2))

    return out

def body():
    out = nn.Sequential()
    for nfilters in [16, 32, 64]:
        out.add(down_sample(num_filters=nfilters))
    return out

# Build a SSD model.
'''
SSD includes 4 modules: 
body network for extracting features.
3 down_sample for generating multi scale features.
5 classification modules.
5 box predicting modules.

The predictions are applied on the main(body) network output, halved module output(down_sample), 
and finally the global pooled layer, respectively.
'''
def ssd_model(num_anchors, num_classes):
    downsamplers = nn.Sequential()
    for _ in range(3):
        downsamplers.add(down_sample(128))
        
    class_predictors = nn.Sequential()
    box_predictors = nn.Sequential()    
    for _ in range(5):
        class_predictors.add(class_predictor(num_anchors, num_classes))
        box_predictors.add(box_predictor(num_anchors))

    model = nn.Sequential()
    model.add(body(), downsamplers, class_predictors, box_predictors)

    return model

# Conpute prediction.
def ssd_forward(x, model, sizes, ratios, verbose=False):    
    body, downsamplers, class_predictors, box_predictors = model
    anchors, class_preds, box_preds = [], [], []
    # anchor box, class prediction, box prediction.

    # feature extraction
    x = body(x)
    for i in range(5):
        # predict
        anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
        class_preds.append(flatten_prediction(class_predictors[i](x)))
        box_preds.append(flatten_prediction(box_predictors[i](x)))

        if verbose:
            print('Predict scale', i, x.shape, 'with', anchors[-1].shape[1], 'anchors')

        # down sample
        if i < 3:
            x = downsamplers[i](x)
        elif i == 3:
            # nn.GlobalMaxPool2D()
            x = nd.Pooling(x, global_pool=True, pool_type="max", kernel=(x.shape[2], x.shape[3]))

    # concat date
    return (concat_predictions(anchors),
            concat_predictions(class_preds),
            concat_predictions(box_preds))

# class SSD, whole model.
class SSD(nn.Block):
    def __init__(self, num_classes, sizes, ratios, verbose=False, **kwargs):
        super(SSD, self).__init__(**kwargs)
        # anchor box sizes and ratios for 5 feature scales. 
        self.sizes = sizes
        self.ratios = ratios
        self.num_classes = num_classes
        self.verbose = verbose
        num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        # use name_scope to guard the names
        with self.name_scope():
            self.model = ssd_model(num_anchors, self.num_classes)

    def forward(self, x):
        anchors, class_preds, box_preds = ssd_forward(
            x, self.model, self.sizes, self.ratios, verbose=self.verbose)
        # it is better to have class predictions reshaped for softmax computation       
        class_preds = class_preds.reshape(shape=(0, -1, self.num_classes+1))
        return anchors, class_preds, box_preds

if __name__ == '__main__':
    from mxnet import nd
    # test network
    sizes = [[.2, .272], [.37, .447], [.54, .619], 
             [.71, .79], [.88, .961]]
    ratios = [[1, 2, .5]] * 5

    net = SSD(num_classes=20, sizes=sizes, ratios=ratios)
    net.initialize()

    x = nd.random.uniform(shape=(1, 3, 300, 300))
    y = net(x)