# import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from encoder import DataEncoder
from retinanet import RetinaNet


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3659,0.3790,0.3179), (0.2910,0.2930,0.2577))
])

inverse_transform = transforms.Compose([
    Denormalize((0.3659,0.3790,0.3179), (0.2910,0.2930,0.2577)),
    transforms.ToPILImage()
])


data_encoder = DataEncoder()
net = RetinaNet()
checkpoint = torch.load('checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

# image_fn = 'demo/session5_left_35.png'
# image_fn = 'demo/session4_right_23.png'
# image_fn = 'demo/session1_right_26.png'
image_dir = os.path.join('/', 'home', 'azzarcher', 'datasets', 'shelf', '6', 'Images')
for image_fn in tqdm(os.listdir(image_dir)):
    frame = Image.open(os.path.join(image_dir, image_fn))
    frame.thumbnail((1536, 864), Image.ANTIALIAS)
    
    # frame = cv2.normalize(frame, frame, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # height, width, _ = frame.shape
    width, height = frame.size
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame).cuda()
    
    # frame = np.expand_dims(frame, 0)
    # frame = Variable(torch.from_numpy(frame).type(torch.FloatTensor), volatile=True).cuda().permute(0, 3, 1, 2)
    frame = Variable(frame.unsqueeze(0))
    loc_preds, cls_preds = net(frame)
    cls_preds = F.softmax(cls_preds.contiguous().squeeze()) # simon was here
    
    loc_preds = loc_preds.data.cpu().squeeze()
    cls_preds = cls_preds.data.cpu().squeeze()
    try:
        boxes, labels = data_encoder.decode(loc_preds, cls_preds, (width, height))
    except RuntimeError:
        frame = inverse_transform(frame.data.cpu().squeeze())
        frame.save(os.path.join('demo', image_fn))
        continue
    
    # frame = frame.permute(0, 2, 3,21).data.cpu().squeeze().numpy()
    # frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    #frame = transforms.ToPILImage()(frame.data.cpu().squeeze())
    frame = inverse_transform(frame.data.cpu().squeeze())
    for box in boxes:
        box = tuple(map(lambda c: int(c), box))
        # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.putText(frame, 'object', (box[0], box[1] - 12), 0, 0.6, (0, 255, 0), 2)
        draw = ImageDraw.Draw(frame)
        #draw.rectangle([box[0], box[1], box[2] + box[0], box[3] + box[1]])
        draw.rectangle([box[0], box[1], box[2], box[3]])
    #cv2.imwrite('/tmp/result.png', frame)
    #cv2.imwrite('result.png', frame)
    frame.save(os.path.join('demo', image_fn))
    #cv2.imshow('result', frame)
    #cv2.waitKey(0)
