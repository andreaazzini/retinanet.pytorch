import os

root = os.path.join('data', 'VOC2012')
image_dir = os.path.join(root, 'JPEGImages')
annotation_dir = os.path.join(root, 'Annotations')
train_imageset_fn = os.path.join(root, 'ImageSets', 'Main', 'trainval.txt')
val_imageset_fn = os.path.join(root, 'ImageSets', 'Main', 'val.txt')
image_ext = '.jpg'

backbone = 'resnet101'
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
scale = None

batch_size = 8
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
num_epochs = 124
lr_decay_epochs = [83, 110]
num_workers = 8

eval_while_training = False
eval_every = 10
