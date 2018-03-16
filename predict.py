# -*- coding:utf-8 -*-
"""
Predict the class and box by giving image.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxDetection
from net import SSD

# Load image data.
def process_image(fname, data_shape):
    with open(fname, 'rb') as f:
        im = mx.image.imdecode(f.read()) # Use mxnet image read function.
    # resize to data_shape
    data = mx.image.imresize(im, data_shape, data_shape)
    # minus rgb mean
    data = data.astype("float32")
    # convert to batch x channel x height x width. batch_size=1.
    return data.transpose((2, 0, 1)).expand_dims(axis=0), im

def box_to_rect(box, color, linewidth=1):
    """
    convert an anchor box to a matplotlib rectangle.
    """
    box = box.asnumpy()
    return plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=linewidth)

def display(im, out, class_names, colors, threshold=0.5):    
    plt.imshow(im.asnumpy())
    for row in out:
        row = row.asnumpy()
        '''
        predict(x)[0] is a 7146 * 6 NDArray, the format of ecah line is:
        class_id class_score xmin ymin xmax ymax
        7146 represent, there are 7146 anchor boxes.

        If class_id is -1, then class_score is small, this represent the box predicted only
        contain the background, or it is dropped.
        '''
        class_id, score = int(row[0]), row[1]
        # print(class_id, score)
        if class_id < 0 or score < threshold:
            continue
        color = colors[class_id % len(colors)]
        box = row[2:6] * np.array([im.shape[0], im.shape[1]] * 2)
        rect = box_to_rect(nd.array(box), color, 2)
        plt.gca().add_patch(rect)
                        
        text = class_names[class_id]
        plt.gca().text(box[0], box[1], 
                       '{:s} {:.2f}'.format(text, score),
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=10, color='white')
    plt.show()

# Predict.
# Every pixel will produce many anchor boxes, so we will get many similar box, result in a bad
# result. So we should use NMS algorithm to suppress some boxes. NMS algorithm has implemented
# in class MultiBoxDetection.
# When generating anchor box NDArray, classification prediction, box prediction, we can use
# this class on these outputs directly. like: MultiBoxDetection(cls_probs, box_preds, anchors).
def predict(x, net, ctx):
    anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
    cls_probs = nd.SoftmaxActivation(
        cls_preds.transpose((0, 2, 1)), mode='channel')

    return MultiBoxDetection(cls_probs, box_preds, anchors, force_suppress=True, clip=False)
    '''
    MultiBoxDetection class, Convert multibox detection predictions. The arguments are:
    cls_prob: NDArray, Class probabilities.
    loc_pred: NDArray, Location regression predictions.
    anchor: NDArray, Multibox prior anchor boxes.
    clip: Clip out-of-boundary boxes. default=True.
    force_suppress: Suppress all detections regardless of class_id. 
    nms_threshold: Non-maximum suppression threshold. default=0.5.

    return NDArray.
    '''

if __name__ == '__main__':
    # anchor box sizes and ratios for 5 feature scales. May have other choices.
    sizes = [[.2, .272], [.37, .447], [.54, .619], 
                  [.71, .79], [.88, .961]]
    ratios = [[1, 2, .5]] * 5
    # [[1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5]].

    # network
    net = SSD(num_classes=20, sizes=sizes, ratios=ratios)
    # There are 21 classes, the first class is background. label form 0 to 20.
    # In the succeeding process, num_classes will plus 1.
    # Load trained parameters.
    net.load_params("./model/SSD_300x300-1.params", ctx=mx.gpu(0))

    '''
    predict() will output all anchor box.
    predict(x)[0] is a 7146 * 6 NDArray, the format of ecah line is:
        class_id class_score xmin ymin xmax ymax
    So, every anchor box is [class_id, confidence, xmin, ymin, xmax, ymax].
    7146 represent, there are 7146 anchor boxes.

    If class_id is -1, then class_score is small, this represent the box predicted only
    contain the background, or it is dropped.
    '''
    x, im = process_image("/home/ly/caffe-ssd/data/VOC0712/VOC2007/JPEGImages/006398.jpg", 
        data_shape=300)
    out = predict(x, net=net, ctx=mx.gpu(0))
    # print(out[0]) # 7146 * 6, NDArray.

    # Set a threshold, if class_score(confidence) > threshold, plot this anchor box on the image.
    mpl.rcParams['figure.figsize'] = (6, 6)
    colors = ['blue', 'green', 'red', 'black', 'magenta']
    class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # Use class_names when predicting the box of giving image.
    display(im, out[0], class_names=class_names, colors=colors, threshold=0.15)