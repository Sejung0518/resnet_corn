# import package
import tensorflow as tf
# model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet
from tensorflow import uint32
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
from torchsummary import summary
import time
import copy

from torchvision.models import ResNet

import config
from bottleneckblock import Bottleneck
from basicblock import BasicBlock
from dataset import Dataset

from torch.optim.lr_scheduler import ReduceLROnPlateau


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


corn_class = ['Infected', 'Healthy']

train_transform = transforms.ToTensor()
valid_transform = transforms.ToTensor()

train_dataset = Dataset(config.train_image_path, train_transform, config.image_format)
valid_dataset = Dataset(config.valid_image_path, valid_transform, config.image_format)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)

writer = SummaryWriter(log_dir="./summary")

# To normalize the dataset, calculate the mean and std
train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in train_dataset]
train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in train_dataset]

train_meanR = np.mean([m[0] for m in train_meanRGB])
train_meanG = np.mean([m[1] for m in train_meanRGB])
train_meanB = np.mean([m[2] for m in train_meanRGB])

train_stdR = np.mean([s[0] for s in train_stdRGB])
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])

val_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in valid_dataset]
val_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in valid_dataset]

val_meanR = np.mean([m[0] for m in val_meanRGB])
val_meanG = np.mean([m[1] for m in val_meanRGB])
val_meanB = np.mean([m[2] for m in val_meanRGB])

val_stdR = np.mean([s[0] for s in val_stdRGB])
val_stdG = np.mean([s[1] for s in val_stdRGB])
val_stdB = np.mean([s[2] for s in val_stdRGB])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet34().to(device)
x = torch.randn(3, 3, 224, 224).to(device)
output = model(x)
# print(output.size())

# summary(model, (3, 224, 224), device=device.type)

loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.001)

lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)


# function to get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# function to calculate metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# function to calculate loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        print("xb:", xb)
        print("type of yb elements:", yb)
        # yb가 아니라 새로운 튜플 만들어서 convert_to_tensor 적용한 요소를 넣어야 할 것 같음
        for ele in yb:
            ele = tf.convert_to_tensor(corn_class.index(ele), dtype=uint32)
            # print(type(ele))
            # ele = tf.convert_to_tensor(ele)
            # print(type(ele))
        print("type of yb elements:", type(yb), type(yb[0]), len(yb))
        yb = torch.tensor(yb)

        # yb = torch.stack(list(yb), dim=0)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric


# function to start training
def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # # GPU out of memoty error
    # best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            # best_model_wts = copy.deepcopy(model.state_dict())

            # torch.save(model.state_dict(), path2weights)
            # print('Copied best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' % (
            train_loss, val_loss, 100 * val_metric, (time.time() - start_time) / 60))
        print('-' * 10)

    # model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history


# define the training parameters
params_train = {
    'num_epochs': 20,
    'optimizer': opt,
    'loss_func': loss_func,
    'train_dl': train_dataloader,
    'val_dl': valid_dataloader,
    'sanity_check': False,
    'lr_scheduler': lr_scheduler,
    'path2weights': './models/weights.pt',
}


# create the directory that stores weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')


createFolder('./models')

model, loss_hist, metric_hist = train_val(model, params_train)
