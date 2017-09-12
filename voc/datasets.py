import os
import random
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset

from voc.annotations import AnnotationDir


class VocLikeDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, imageset_fn, image_ext, classes, encoder, transform=None, val=False):
        self.image_dir_path = image_dir
        self.image_ext = image_ext
        with open(imageset_fn) as f:
            self.filenames = [fn.rstrip() for fn in f.readlines()]
        self.annotation_dir = AnnotationDir(annotation_dir, self.filenames, classes, '.xml', 'voc')
        self.filenames = list(self.annotation_dir.ann_dict.keys())
        self.encoder = encoder
        self.transform = transform
        self.val = val

    def __getitem__(self, index):
        fn = self.filenames[index]
        image_fn = '{}{}'.format(fn, self.image_ext)
        image_path = os.path.join(self.image_dir_path, image_fn)
        image = Image.open(image_path)
        boxes = self.annotation_dir.get_boxes(fn)
        example = {'image': image, 'boxes': boxes}
        if self.transform:
            example = self.transform(example)
        return example

    def __len__(self):
        return len(self.filenames)

    def collate_fn(self, batch):
        imgs = [example['image'] for example in batch]
        boxes  = [example['boxes'] for example in batch]
        labels = [example['labels'] for example in batch]
        img_sizes = [img.size()[1:] for img in imgs]

        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i,:,:imh,:imw] = im

            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(max_w, max_h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if not self.val:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets)
        return inputs, img_sizes, torch.stack(loc_targets), torch.stack(cls_targets)
