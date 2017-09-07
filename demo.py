import cv2
import numpy as np
# from PIL import Image

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from encoder import DataEncoder
from retinanet import RetinaNet


data_encoder = DataEncoder()
net = RetinaNet()
checkpoint = torch.load('checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

image_fn = 'demo/session5_left_35.png'

frame = cv2.imread(image_fn)
print(frame.shape)
frame = cv2.normalize(frame, frame, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
height, width, _ = frame.shape
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = np.expand_dims(frame, 0)
frame = Variable(torch.from_numpy(frame).type(torch.FloatTensor), volatile=True).cuda().permute(0, 3, 1, 2)
loc_preds, cls_preds = net(frame)
print(cls_preds, type(cls_preds), cls_preds.size())
cls_preds = F.sigmoid(cls_preds)
cls_preds = cls_preds.clamp(1e-7, 1. - 1e-7)

# cls_preds = F.softmax(cls_preds.contiguous().squeeze()) # simon was here

loc_preds = loc_preds.data.cpu().squeeze()
cls_preds = cls_preds.data.cpu().squeeze()
print(loc_preds)
print(cls_preds)
boxes, labels = data_encoder.decode(loc_preds, cls_preds, (height, width))

frame = frame.permute(0, 2, 3, 1).data.cpu().squeeze().numpy()
frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
for box in boxes:
    box = tuple(map(lambda c: int(c), box))
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(frame, 'object', (box[0], box[1] - 12), 0, 0.6, (0, 255, 0), 2)
#cv2.imwrite('/tmp/result.png', frame)
cv2.imwrite('result.png', frame)
#cv2.imshow('result', frame)
#cv2.waitKey(0)
