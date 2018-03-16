# -*- coding:utf-8 -*-
"""
Loading data. For object detection task, we will utlize mxnet.image.ImageIter() to load Record file.
mxnet.image.ImageDetIter() is a class for object detection task, it's similar to mxnet.image.ImageIter.
mxnet.image.ImageIter will return the a label of one image.
mxnet.image.ImageDetIter() will return the labels of all objects in image and their bounding box.
"""
import os
import mxnet as mx

def get_iterators(args):
    # We will define class_names outside the get_iterators() function.
    # get_iterators() function only return the train and val data iter.
    # Since Detection task is sensitive to object localization, any modification to image that introduced 
    # localization shift will require correction to label, and a list of augmenters specific for Object 
    # detection is provided. So, in this experiment, we will not modify thedata.

    # header width is 2, header width is corresponding to the lst file format.
    # Header width is 2: head + label width.
    # Header width is 4: head + label width + image width + image height.
    train_iter = mx.image.ImageDetIter(
        batch_size = args.batch_size,
        data_shape = args.data_shape,
        path_imgrec = os.path.join(args.data_dir, "trainval.rec"), 
        path_imgidx = os.path.join(args.data_dir, "trainval.idx"), # pikachu_train, trainval
        shuffle = True)
    # For training data, we can transfer *.idx. In this way, the process of shuffle is better.
    # You can see this discussion on https://discuss.gluon.ai/t/topic/3331/6. The seed is not fixed when
    # shuffle training data.

    val_iter = mx.image.ImageDetIter(
        batch_size = args.batch_size,
        data_shape = args.data_shape,
        path_imgrec = os.path.join(args.data_dir, "val.rec"), # pikachu_val, val
        shuffle = False)
    # For validating data, shuffle is False.

    return train_iter, val_iter

def box_to_rect(box, color, linewidth=1):
    """
    convert an anchor box to a matplotlib rectangle.
    """
    box = box.asnumpy()
    return plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=linewidth)

if __name__ == '__main__':
    # When we call the main function, we would load these modules.
    import argparse
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mxnet import nd
    # parse args
    parser = argparse.ArgumentParser(description="Loading object detection data iter.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(
        # data
        data_dir = "./datasets/pikachu", # We transfer data_dir directly.
        batch_size = 16,
        data_shape = (3, 256, 256) # 300 * 300.
        )
    args = parser.parse_args()

    train_data, val_data = get_iterators(args)
    # Now, we will load a batch size data. The train_data is a object, i.e. a data iter, you can call 
    # next() to generate the numpy ndarray.
    batch = train_data.next()
    train_data.reset()
    for i, _ in enumerate(train_data):
        print(i)
    # print(batch)
    # Use batch.data() to get NDArray.
    '''
    Generate a batch size data, like:
    trainval.rec: DataBatch: data shapes: [(4L, 3L, 300L, 300L)] label shapes: [(4L, 56L, 6L)].
    val.rec: DataBatch: data shapes: [(4L, 3L, 300L, 300L)] label shapes: [(4L, 42L, 6L)].
    The shape of label is: [batch_size, num_object_per_image, label_width].
    num_object_per_image is the max number of object in all images.

    56 and 42 can be generated like:
        f = open("/home/ly/mxnet/example/ly/SSD-gulon/datasets/VOC0712/val.lst")
        a = []
        for b in f.readlines():
            a.append(b.replace("\t", ","))

        num = []
        for c in a:
            num.append(len(c.split(",")))

        print(max(num)) # 256. 256 - 4 = 252, since each line has: index, head_width, label_width, image_path
        # 252/6 = 42.
    56 and 42 is the max number of object in all training and valing images.
    '''

    '''
    # Now, we want show some images and its label info. 
    mpl.rcParams['figure.dpi']= 120
    _, figs = plt.subplots(3, 3, figsize=(6, 6))
    data_shape = 300

    # batch.data[0] is 4 * 3 * 300 * 300 NDArray, in mxnet it will skip the first dim. 
    # So, batch.data[0][i] is 3 * 300 * 300 NDArray, i=0:batch_size-1.
    # It will show 3*3 images and their labels rectangle.
    for i in range(3):
        for j in range(3):
            img, labels = batch.data[0][3 * i + j], batch.label[0][3 * i + j]
            # print(batch.data[0][3 * i + j]) # 3 * 300 * 300 NDArray, i=0:batch_size-1.
            # (3L, 300L, 300L) => (300L, 300L, 3L)
            img = img.transpose((1, 2, 0))
            img = img.clip(0, 255).asnumpy() / 255.
            fig = figs[i][j] # show a subplot.
            fig.imshow(img)

            for label in labels:
                rect = box_to_rect(label[1:5] * data_shape, "red", 2)
                fig.add_patch(rect)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

    plt.show()
    '''