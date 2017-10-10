import os

root = os.path.join('data', 'food-collage')
image_dir = os.path.join(root, 'JPEGImages')
annotation_dir = os.path.join(root, 'Annotations')
train_imageset_fn = os.path.join(root, 'ImageSets', 'Main', 'trainval.txt')
val_imageset_fn = os.path.join(root, 'ImageSets', 'Main', 'trainval.txt')
image_ext = '.jpg'
image_dir = os.path.join('/', 'datasets', 'miele_testset_traderjoes') # TEST

backbone = 'resnet50'
classes = ['onion', 'rice', 'raw_zucchini', 'apple', 'cucumber', 'bell_pepper', 'asparagus', 'potato', 'champignon', 'ginger', 'tomato', 'garlic', 'lettuce', 'carrot', 'broccoli', 'celery', 'lemon', 'raw_chicken_breast', 'raw_salmon_fillet', 'raw_pork_tenderloin', 'cauliflower', 'egg', 'kohlrabi', 'flour', 'plum']
mean, std = (0.3659,0.3790,0.3179), (0.2910,0.2930,0.2577)
scale = None
scale = (341, 256) # TEST

batch_size = 32
batch_size = 1 # TEST
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
num_epochs = 9000
lr_decay_epochs = [6000, 8000]
num_workers = 8
eval_every = 10
