from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)
    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.
    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    return mask.scatter_(1, index, ones)

class FocalLoss(nn.Module):
    num_classes = 2
    # num_classes = 21

    def __init__(self):
        super(FocalLoss, self).__init__()

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        #print(y)
        y = one_hot(y.cpu(), x.size(-1)).cuda()
        logit = F.softmax(x)
        #logit = F.sigmoid(x)
        logit = logit.clamp(1e-7, 1. - 1e-7)

        loss = -1 * y.float() * torch.log(logit)
        loss = loss * (1 - logit) ** 2
        return loss.sum()
 
#         y_bg = (y == 0).float()
#         y_fg = (y > 0).float()
# 
#         # y_bg = Variable(y_bg)
#         # y_fg = Variable(y_fg)
# 
#         #y_bg = Variable(torch.stack((y_bg, y_bg), 1))
#         #y_fg = Variable(torch.stack((y_fg, y_fg), 1))
# 
#         #y = Variable(y) # batch_size x bboxes
#         # x : batch_size x bboxes x classes
# 
#         alpha = 0.25
#         gamma = 2
#         x = x[:, 1]
#         p = F.sigmoid(x)
#         p_t = p * y_fg + (1 - p) * y_bg +0.000001
#         a_t = alpha * y_fg + (1 - alpha) * y_bg
#         fl = -a_t*(1-p_t)**gamma * p_t.log()
#         return fl.sum()

        # alpha = 0.75
        # gamma = 2
        # #logp = F.log_softmax(x)
        # logp = F.logsigmoid(x)
        # p = logp.exp()
        # w = alpha*(y>0).float() + (1-alpha)*(y==0).float()
        # wp = w.view(-1,1) * (1-p).pow(gamma) * logp
        #return F.nll_loss(wp, y, size_average=False)

    def focal_loss_alt(self,x,y):
        '''Focal loss alternate described in appendix.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.75
        gamma = 2
        beta = 1
        x[y.view(-1,1).expand_as(x)==0] *= -1
        w = alpha*(y>0).float() + (1-alpha)*(y==0).float()
        wp = w.view(-1,1)*F.log_softmax(gamma*x+beta)
        return F.nll_loss(wp, y, size_average=False) / gamma

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        # mask = pos_neg.unsqueeze(2).expand_as(loc_preds) # andrea was here
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        # mask = pos.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        #masked_cls_preds = cls_preds[mask].view(-1, self.num_classes-1)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        # cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos])

        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        loss = (loc_loss+cls_loss)/num_pos
        return loss
