import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional
import numpy as np
import os
import copy
import argparse
import random
import test
import torch.nn.utils.prune as prune
from copy import deepcopy
import copy
import wandb
from train import loss_L1_fft as losses
import time
from torch.utils.data import Dataset, DataLoader
from data.data_RGB import get_training_data
from tensorboardX import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from model.MSSNet import DeblurNet as teacher
from model.MSSNet_pruned import DeblurNet as student
from train.trainer_dp import Trainer

WANDB_KEY = os.environ['WANDB_KEY']
device0 = torch.device("cuda:1")
device1 = torch.device("cuda:0")
lr_initial = .001
lr_min = .0000001
max_epoch = 600
crop_size = 256
train_datalist = './datalist/datalist_gopro_train.txt'
data_dir = './dataset/GOPRO_Large/train'
teacher_hook_list = []
student_hook_list = []
teacher_layers = []
student_layers = []

def cosLoss(student, teacher):
    sim = nn.functional.cosine_similarity(student.flatten(start_dim=1), teacher.flatten(start_dim=1)).abs().mean()
    return 1-sim

def teacherHooks(layer_list, layer_num):
    def hook(module,input, output):
        global teacher_layers
        non_zero_ind = torch.nonzero(module.weight[:, 0, 0, 0]).flatten()
        teacher_layers.append(output[:,non_zero_ind,:,:])
    return hook

def studentHooks(layer_list, layer_num):
    def hook(module,input, output):
        global student_layers
        student_layers.append(output)
    return hook

def apply_hooks(model, hook_fn, layer_list, hook_list):
    num = 0
    for layer in model.modules():
        if str(type(layer)).split('.')[-1][:-2] == 'ResBlock':
            act = list(layer.named_modules())[2][1]
            hook_list.append(act.register_forward_hook(hook_fn(layer_list, num)))
            num+=1

def save_model(model, optimizer, epoch, scheduler):
    global student_hook_list
    for hook in student_hook_list:
        hook.remove()
    file_name = './student_models/student_{}.pth'.format(int(time.time()))
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch}, file_name)
    student_hook_list = []
    apply_hooks(model, studentHooks, student_layers, student_hook_list)


def train(teacher_model, student_model, train_dataloader,optim,scheduler,scaler, l1loss, fftloss):
    train_batch_num = len(train_dataloader)
    start_epoch = time.time()
    epoch_loss = 0
    student_model.train()
    teacher_model.eval()

    for iteration, data in enumerate(train_dataloader):

        # zero_grad #########################
        for param in student_model.parameters():
            param.grad = None
        for param in teacher_model.parameters():
            param.grad = None
        ####################################

        global student_layers, teacher_layers
        blur_images_teacher = data[1].to(device0)
        blur_images_student = data[1].to(device1)
        with torch.cuda.amp.autocast():
            student_out = student_model(blur_images_student)
            with torch.no_grad():
                teacher_out = teacher_model(blur_images_teacher)
            #layer_loss = sum([cosLoss(student_layers[i],teacher_layers[i].to(device1)).item() for i in range(50,87)])
            #scaler.scale(layer_loss).backward(retain_graph=True)
            #layer_loss = 0
            #for j in range(0, len(student_layers), 5):
            #    layer_loss += cosLoss(student_layers[j], teacher_layers[j].to(device1))
            #for i in range(len(student_out)-1):
            #    layer_loss  += cosLoss(student_out[i],teacher_out[i].to(device1)).item()
            loss_l1 = sum([l1loss(student_out[j], teacher_out[j].to(device1)) for j in range(len(student_out))])
            loss_fft = sum([fftloss(student_out[j], teacher_out[j].to(device1)) for j in range(len(student_out))])
            loss = (loss_l1) + (0.1 * loss_fft)
        epoch_loss+= loss.item()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        epoch_loss += loss.item()
        student_layers = []
        teacher_layers = []
        #epoch_loss += loss.item()
    print('Total Epoch Time: {}'.format(int(time.time()-start_epoch)))
    scheduler.step()
    print(scheduler.get_last_lr())
    return epoch_loss

def main():
    wandb.login(key=WANDB_KEY)
    teacher_model = teacher(wf=54, scale=42, vscale=42).to(device0)
    teacher_model.load_state_dict(torch.load('./checkpoint/prune_state.pth')['model_state_dict'])
    student_model = student(wf=54//3, scale=42//3, vscale=42//3).to(device1)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr_initial, betas=(0.9, 0.999),eps=1e-8)
    #optimizer = torch.optim.SGD(student_model.parameters(), lr=lr_initial)
    l1loss = losses.L1Loss()
    fftloss = losses.FFTLoss()
    warmup_epochs = 3
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch, lr_min)
    #scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    #scheduler.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5, verbose=True)
    train_dataset = get_training_data(train_datalist, data_dir, {'patch_size': crop_size})
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4,
                              drop_last=False, pin_memory=True)
    checkpoint = torch.load('./student_models/student_1670353965.pth')
    last_epoch = checkpoint['epoch']
    #last_epoch = 0
    student_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    #apply_hooks(teacher_model, teacherHooks, teacher_layers, teacher_hook_list)
    #apply_hooks(student_model, studentHooks, student_layers, student_hook_list)
    run = wandb.init(
        name='Distillation',
        reinit=True,
        #resume="must",
        #id='3lk05jl0',
        entity='f22idl',
        project="deblur"
    )
    for epoch in range(max_epoch):
        curr_epoch = last_epoch + epoch
        print('Epoch: {}'.format(curr_epoch+1))
        train_loss = train(teacher_model, student_model, train_loader,optimizer,scheduler,scaler, l1loss, fftloss)
        print('Training Loss: {}'.format(train_loss))
        wandb.log({"Training Loss": train_loss})
        if (curr_epoch + 1)%25 == 0:
            print('Saving Model')
            save_model(student_model, optimizer, epoch+1, scheduler)
    run.finish()
if __name__ == '__main__':
    main()