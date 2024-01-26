#
# Note -- this training script is tweaked from the original version at:
#
#           https://github.com/pytorch/vision/tree/v0.3.0/references/segmentation
#
# It's also meant to be used against this fork of torchvision, which includes 
# some patches for exporting to ONNX and adds fcn_resnet18 and fcn_resnet34:
#
#           https://github.com/dusty-nv/vision/tree/v0.3.0
#
import argparse
import datetime
import time
import os
from pathlib import Path
import torch
import torch.utils.data
from torch import nn
import torchvision
from collections import namedtuple
from typing import Any
import dataset
import transforms as T
import utils
try:
    from torch2trt import TRTModule
    ON_JETSON = True
    from tensorboard_dummy import SummaryWriter
except:
    ON_JETSON = False
    from torch.utils.tensorboard import SummaryWriter


model_names = sorted(name for name in torchvision.models.segmentation.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.segmentation.__dict__[name]))

#
# parse command-line arguments
#
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fcn_resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: fcn_resnet18)')
    parser.add_argument('--aux-loss', action='store_true', help='train with auxilliary loss')
    parser.add_argument('--resolution', default=320, type=int, metavar='N',
                        help='NxN resolution used for scaling the training dataset (default: 320x320) '
                         'to specify a non-square resolution, use the --width and --height options')
    parser.add_argument('--width', default=argparse.SUPPRESS, type=int, metavar='X',
                        help='desired width of the training dataset. if this option is not set, --resolution will be used')
    parser.add_argument('--height', default=argparse.SUPPRESS, type=int, metavar='Y',
                        help='desired height of the training dataset. if this option is not set, --resolution will be used')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--model-dir', default='.', help='path where to save output models')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
   
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


#
# create data transform
#
def get_transform(train, resolution):
    transforms = []

    # if square resolution, perform some aspect cropping
    # otherwise, resize to the resolution as specified
    if resolution[0] == resolution[1]:
        base_size = resolution[0] + 32 #520
        crop_size = resolution[0]      #480

        min_size = int((0.5 if train else 1.0) * base_size)
        max_size = int((2.0 if train else 1.0) * base_size)

        transforms.append(T.RandomResize(min_size, max_size))

        # during training mode, perform some data randomization
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.RandomCrop(crop_size))
    else:
        transforms.append(T.Resize(resolution))

        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


#
# define the loss functions
#
def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


#
# evaluate model IoU (intersection over union)
#
def evaluate(model, data_loader, device, num_classes, on_jetson):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if not on_jetson:
                output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten(), on_jetson)
            
        confmat.reduce_from_all_processes()

    return confmat


#
# train for one epoch over the dataset
#
def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, tb_writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    iteration = 1
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()
        tb_writer.add_scalar("loss", loss.item(), epoch * 100 + iteration)
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        iteration += 1
class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor, dummy) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data



#
# main training function
#
def main(args):
    if args.data:
        utils.mkdir(Path("models") / args.data)

    utils.init_distributed_mode(args)
    print(args)
    tb_writer = SummaryWriter(Path("runs") / args.data)
    tb_writer.add_text("Args", str(args))
    device = torch.device(args.device)

    # determine the desired resolution
    resolution = (args.resolution, args.resolution)

    if "width" in args and "height" in args:
        resolution = (args.height, args.width)     
    
    # load the train and val datasets
    dataset_train = dataset.get_dataset("train", get_transform(train=True, resolution=resolution), ON_JETSON)
    dataset_test = dataset.get_dataset("val", get_transform(train=False, resolution=resolution), ON_JETSON)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True, persistent_workers=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, persistent_workers=True)

    print("=> training with dataset: (train={:d}, val={:d})".format(len(dataset_train), len(dataset_test)))
    print("=> training with resolution: {:d}x{:d}".format(resolution[1], resolution[0]))
    print("=> training with model: {:s}".format(args.arch))
    num_classes = 21
    # create the segmentation model
    if not ON_JETSON:
        model = torchvision.models.segmentation.__dict__[args.arch](num_classes=num_classes,
                                                                    aux_loss=args.aux_loss,
                                                                    weights=None # do not use pretrained weights
                                                                    )

        # Modify the final classification layer for two classes
        in_channels = model.classifier[4].in_channels
        new_classifier = torch.nn.Conv2d(in_channels, 2, kernel_size=1)  # 2 output channels for 2 classes
        model.classifier[4] = new_classifier

        item = next(iter(data_loader))
        tb_model = ModelWrapper(model)
        tb_writer.add_graph(tb_model, item)
        model.to(device)
    else:
        #ON_JETSON
        model = TRTModule()
        model.load_state_dict(torch.load(f'./models/{args.arch}_trt.pth'))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # eval-only mode
    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, on_jetson=ON_JETSON)
        print(confmat)
        return

    # create the optimizer
    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]

    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # training loop
    start_time = time.time()
    best_IoU = 0.0

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train the model over the next epoc
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, tb_writer)

        # test the model on the val dataset
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, on_jetson=ON_JETSON)
        print(confmat)

        if confmat.mean_IoU > best_IoU:
            best_IoU = confmat.mean_IoU
            checkpoint_path = Path("models") / args.data / 'best_model.pth'
            if Path.is_file(checkpoint_path):
                os.remove(checkpoint_path)
            utils.save_on_master(
            {
                'model': model_without_ddp,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'arch': args.arch,
                'num_classes': num_classes,
                'resolution': resolution,
                'accuracy': confmat.acc_global,
                'mean_IoU': confmat.mean_IoU
            },
            checkpoint_path)
            print('saved best model to:  {:s}  ({:.3f}% mean IoU, {:.3f}% accuracy)'.format(str(checkpoint_path), best_IoU, confmat.acc_global))
        
        tb_writer.add_scalar(f'accuracy', confmat.acc_global, epoch)
        tb_writer.add_scalar(f'mean_IoU', confmat.mean_IoU, epoch)
        tb_writer.add_text(f'Weights', str(checkpoint_path))

        tb_writer.flush()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    tb_writer.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)

