parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--exp', required=True, help='experiment name')
args = parser.parse_args()

sys.path.insert(0, os.path.join('exps', args.exp))
import config as cfg


transform_list = [transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
if cfg.scale is not None:
    transform_list.insert(0, transforms.Scale(cfg.scale))
transform = transforms.Compose(transform_list)

data_encoder = DataEncoder()
valset = VocLikeDataset(image_dir=cfg.image_dir, annotation_dir=cfg.annotation_dir, imageset_fn=cfg.val_imageset_fn,
                        image_ext=cfg.image_ext, classes=cfg.classes, encoder=data_encoder, transform=val_transform, val=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False,
                                        num_workers=cfg.num_workers, collate_fn=valset.collate_fn)

print('Building model...')
net = RetinaNet(backbone=cfg.backbone, num_classes=len(cfg.classes))
checkpoint = torch.load(os.path.join('ckpts', args.exp, 'ckpt.pth'))
net.load_state_dict(checkpoint['net'])
epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()
cudnn.benchmark = True

net.eval()
ap_acc = 0

for batch_idx, (inputs, input_sizes, loc_targets, cls_targets) in enumerate(valloader):
    inputs = Variable(inputs.cuda())
    loc_targets = Variable(loc_targets.cuda())
    cls_targets = Variable(cls_targets.cuda())
    loc_preds, cls_preds = net(inputs)

    height, width = frame.size()[1:] # TODO I need these to decode. Should I return the from the dataloader?
    
    frame = Variable(frame.unsqueeze(0))
    loc_preds, cls_preds = net(frame)
    cls_preds = F.softmax(cls_preds.contiguous().squeeze())
    
    loc_preds = loc_preds.data.cpu().squeeze()
    cls_preds = cls_preds.data.cpu().squeeze()
    boxes, labels = data_encoder.decode(loc_preds, cls_preds, (width, height)) # TODO update the decoder so that it decodes batches
    ground_truth_boxes = # TODO

    tp = 0
    num_pos = len(boxes)
    num_true = len(ground_truth_boxes)
    for i, box in enumerate(boxes):
        for ground_truth_box in ground_truth_boxes:
            if box.overlaps_with(ground_truth_box):
                tp += 1
                ground_truth_boxes.remove(ground_truth_box)
                break
    precision = tp / num_pos
    recall = tp / num_true
    ap_acc += average_precision(precision, recall)
mean_average_precision = ap_acc / num_images
print('mAP ({}, epoch {}): {}'.format(args.exp, epoch, mean_average_precision))
