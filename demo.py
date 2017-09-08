import argparse
import numpy as np
import os
import sys
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from encoder import DataEncoder
from retinanet import RetinaNet
from voc.transforms import Unnormalize


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--exp', required=True, help='experiment name')
args = parser.parse_args()

sys.path.insert(0, os.path.join('exps', args.exp))
import config as cfg


transform_list = [transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
if cfg.scale is not None:
    transform_list.insert(0, transforms.Scale(cfg.scale))
transform = transforms.Compose(train_transform_list)

inverse_transform = transforms.Compose([
    Unnormalize(cfg.mean, cfg.std),
    transforms.ToPILImage()
])

data_encoder = DataEncoder()
net = RetinaNet(backbone=cfg.backbone, num_classes=len(cfg.classes))
checkpoint = torch.load(os.path.join('ckpts', args.exp, 'ckpt.pth'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

for image_fn in tqdm(os.listdir(cfg.image_dir)):
    frame = Image.open(os.path.join(cfg.image_dir, image_fn))
    frame = transform(frame).cuda()
    height, width = frame.size()[1:]
    
    frame = Variable(frame.unsqueeze(0))
    loc_preds, cls_preds = net(frame)
    cls_preds = F.softmax(cls_preds.contiguous().squeeze())
    
    loc_preds = loc_preds.data.cpu().squeeze()
    cls_preds = cls_preds.data.cpu().squeeze()
    try:
        boxes, labels = data_encoder.decode(loc_preds, cls_preds, (width, height))
    except RuntimeError:
        frame = inverse_transform(frame.data.cpu().squeeze())
        frame.save(os.path.join('demo', image_fn))
        continue
    
    frame = inverse_transform(frame.data.cpu().squeeze())
    for i, box in enumerate(boxes):
        box = tuple(map(lambda c: int(c), box))
        draw = ImageDraw.Draw(frame)
        draw.rectangle([box[0], box[1], box[2], box[3]])
        draw.text((box[0] + 10, box[1] + 10), cfg.classes[labels[i] - 1])
    frame.save(os.path.join('demo', image_fn))
