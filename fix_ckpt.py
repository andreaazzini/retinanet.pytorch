import torch
import os

checkpoint = os.path.join('ckpts', 'voc', 'ckpt.pth')
old_checkpoint = torch.load(os.path.join('ckpts', 'voc', 'ckpt.old.pth'))
net = old_checkpoint['net']
loss = old_checkpoint['loss']
epoch = old_checkpoint['epoch']
lr = 0.01
state = {
    'net': net,
    'loss': loss,
    'epoch': epoch,
    'lr': lr
}
torch.save(state, checkpoint)
