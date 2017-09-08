import numpy as np
import os
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from encoder import DataEncoder
from retinanet import RetinaNet, resnet50_features
from voc.transforms import Unnormalize


transform = transforms.Compose([
    # transforms.Scale((1536, 864)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.3659, 0.3790, 0.3179), (0.2910, 0.2930, 0.2577))
])

inverse_transform = transforms.Compose([
    Unnormalize((0.3659, 0.3790, 0.3179), (0.2910, 0.2930, 0.2577)),
    transforms.ToPILImage()
])


data_encoder = DataEncoder()
net = RetinaNet(resnet50_features())
checkpoint = torch.load('checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

image_fn_1 = 'session5_left_35.png'
image_fn_2 = 'session4_right_23.png'
image_fn_3 = 'session1_right_26.png'
image_dir = os.path.join('/', 'home', 'azzarcher', 'datasets', 'shelf', '6', 'Images')
# for image_fn in tqdm(os.listdir(image_dir)):
for image_fn in tqdm([image_fn_1, image_fn_2, image_fn_3]):
    frame = Image.open(os.path.join(image_dir, image_fn))
    frame.thumbnail((1536, 864), Image.ANTIALIAS)
    
    width, height = frame.size
    frame = transform(frame).cuda()
    
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
    for box in boxes:
        box = tuple(map(lambda c: int(c), box))
        draw = ImageDraw.Draw(frame)
        draw.rectangle([box[0], box[1], box[2], box[3]])
    frame.save(os.path.join('demo', image_fn))
