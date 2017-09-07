import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from datagen import ListDataset
from loss import FocalLoss
from retinanet import RetinaNet, resnet50_features


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3659,0.3790,0.3179), (0.2910,0.2930,0.2577))
    # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(root='../../datasets/shelf/6/Images',
                       list_file='../../datasets/shelf/train.txt', train=True, transform=transform, input_size=864, max_size=1536)
#trainset = ListDataset(root='/datasets/pascal-voc/VOCdevkit/VOC2007/JPEGImages',
#                       list_file='voc_data/voc07_train.txt', train=True, transform=transform, input_size=600, max_size=1000)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=6, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

#testset = ListDataset(root='../../datasets/shelf/6/Images',
#                      list_file='../../datasets/shelf/train.txt', train=False, transform=transform, input_size=864, max_size=1536)
#testset = ListDataset(root='/datasets/pascal-voc/VOCdevkit/VOC2007/JPEGImages',
#                      list_file='voc_data/voc07_test.txt', train=False, transform=transform, input_size=600, max_size=1000)
#testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

net = RetinaNet(resnet50_features(pretrained=True))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()
cudnn.benchmark = True

criterion = FocalLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), max_norm=1.2)
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx+1)))
    save_checkpoint(train_loss, len(trainloader))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss

def save_checkpoint(loss, n):
    global best_loss
    loss /= n
    if loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = loss


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    # test(epoch)
