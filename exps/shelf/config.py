import os

root = os.path.join('data', 'shelf-6')
image_dir = os.path.join(root, 'Images')
annotation_dir = os.path.join(root, 'Annotations')
train_imageset_fn = os.path.join(root, 'ImageSets', 'trainval.txt')
val_imageset_fn = os.path.join(root, 'ImageSets', 'trainval.txt')
image_ext = '.png'

backbone = 'resnet50'
# classes = ['jaya', 'tejava', 'soylent', 'fiberone', 'pocky', 'seaweed', 'cokezero']
classes = ['object']
mean, std = (0.3659,0.3790,0.3179), (0.2910,0.2930,0.2577)
scale = (1536, 864)

batch_size = 2
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
num_epochs = 50
lr_decay_epochs = [40]
num_workers = 8
eval_every = 10
eval_while_training = False
