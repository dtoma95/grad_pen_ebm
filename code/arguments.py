import torch
import torchvision
from torch.utils.data import Dataset
from os.path import join
from torch.utils.data import DataLoader


import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import os
import json

import argparse
import sys
from tqdm import tqdm
import pytorch_lightning as pl
import random

from logger import Logger
from trainer import Trainer
import numpy as np
import math
from sam import SAM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_work", type=str, default='../')
    parser.add_argument("--load_model", type=str, default='')
    parser.add_argument("--load_options", type=str, default='')

    parser.add_argument("--name_model", type=str, default='simple') #'diffgan'
    parser.add_argument("--name_dataset", type=str, default='MNIST') #CIFAR10, MNIST
    parser.add_argument("--use_subset", type=bool, default=False)
    parser.add_argument("--use_subset_label", type=int, default=1)
    parser.add_argument("--num_data_fake", type=int, default=100) # 0 means num_data_fake = num_data_real 
    parser.add_argument("--num_data_real", type=int, default=-1) # -1
    parser.add_argument("--in_memory_dataset", type=bool, default=False)
    

    parser.add_argument("--channel_data", type=int, default=3)
    parser.add_argument("--height_data", type=int, default=32)
    parser.add_argument("--width_data", type=int, default=32)
    parser.add_argument("--device_cuda", type=int, default=1)
    parser.add_argument("--size_batch", type=int, default=100)
    parser.add_argument("--accumulate_iters", type=int, default=1)
    

    parser.add_argument("--length_epoch", type=int, default=70000)
    parser.add_argument("--length_time", type=int, default=300)
    parser.add_argument("--length_langevin", type=int, default=10)
    parser.add_argument("--length_update", type=int, default=2) 

    parser.add_argument("--lr_energy", type=float, default=0.001)
    parser.add_argument("--lr_fake", type=float, default=0.025)
    parser.add_argument("--fake_train_jumps", type=int, default=1)
    parser.add_argument("--do_jump_scaling", type=bool, default=False)
    parser.add_argument("--jump_scaling_div", type=int, default=2)
    parser.add_argument("--random_jumps", type=bool, default=False)
    

    parser.add_argument("--lr_langevin", type=float, default=0.0)#0.00001
    parser.add_argument("--weight_gradient_penalty", type=float, default=0.01)
    parser.add_argument("--weight_total_variation", type=float, default=0.0)
    parser.add_argument("--option_loss", type=str, default='cd')
    parser.add_argument("--penalty_type", type=str, default='interpolate') # interpolate, real_and_fake, clip, squares, none

    parser.add_argument("--option_optim", type=str, default='adamw') # adamw, adam, adamsn, sgd , adamwsn
    parser.add_argument("--weight_decay", type=float, default=0.01) #0.05, 2
    
    parser.add_argument("--beta1", type=float, default=0.9) # 0.9 0.5 0
    parser.add_argument("--beta2", type=float, default=0.999)# 0.999 0.9 0.999
    parser.add_argument("--model_update_num", type=int, default=1) # MODEL UPDATE
    parser.add_argument("--fake_update_num", type=int, default=1)
    parser.add_argument("--mixed_precision", type=bool, default=True) # True
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_start", type=int, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--dim_feature", type=int, default=8)
    parser.add_argument("--dim_output", type=int, default=1)
    parser.add_argument("--dim_encoding", type=int, default=100)

    parser.add_argument("--noisy_real", type=bool, default=False)
    parser.add_argument("--use_conditioning_label", type=bool, default=False)
    parser.add_argument("--clamp_fake", type=bool, default=True)

    parser.add_argument("--use_time", type=bool, default=True)
    parser.add_argument("--use_time_coefficient", type=bool, default=True)
    parser.add_argument("--coefficient_type", type=str, default="linear") #linear, sigmoid6, sigmoid12, sigmoid18, none, cosine, cosine05
    parser.add_argument("--use_time_weighting", type=bool, default=True) #good
    parser.add_argument("--weighting_type", type=str, default="resampler") # resampler_norm, resampler_norm_new, resampler, linear, logistic, none
    parser.add_argument("--anneling_fake", type=bool, default=False)

    parser.add_argument("--validation_freq", type=int, default=2500)
    
    parser.add_argument("--print_freq", type=int, default=2500)
    parser.add_argument("--inference_at_t_minus_2", type=bool, default=False) # THis adds an extra final timestep only during training
    parser.add_argument("--compute_fid", type=bool, default=True)
    parser.add_argument("--fake_data_warmup_at", type=int, default=-1)
    parser.add_argument("--fake_data_warmup_every", type=int, default=-1) #

    parser.add_argument("--noise_start", type=float, default=0.00001)
    parser.add_argument("--noise_end", type=float, default=0.02)
    parser.add_argument("--init_from", type=str, default="gaussian") # informed, gaussian 

    parser.add_argument("--refresh_rate", type=float, default=1) # 1, 0.05
    parser.add_argument("--refresh_rate2", type=float, default=1)
    parser.add_argument("--init_at_time", type=int, default=0)
    
    parser.add_argument("--batch_type", type=str, default='mixed_random') # mixed, none, mixed_random, mixed_random_jumps
    parser.add_argument("--update_order", type=str, default='model_first') #fake_first, model_first, fake_model_time
    parser.add_argument("--buffer_type", type=str, default='normal') #replay_buffer, normal
    parser.add_argument("--refresh_every", type=int, default=1)

    parser.add_argument("--scheduler_type", type=str, default='multi_step') # step, lambda, multi_step, cyclic, linear, none
    parser.add_argument("--scheduler_arg_step", type=int, default=90000) 
    parser.add_argument("--scheduler_gamma", type=float, default=0.0001)
    parser.add_argument("--scheduler_arg_1", type=float, default=0.0002) # cyclic_max
    parser.add_argument("--scheduler_arg_2", type=float, default=0.00005) # cyclic_minn
    parser.add_argument("--world_size", type=int, default=1) 

    parser.add_argument("--logistic_k", type=float, default=12) # 12

    parser.add_argument("--momentum_gamma", type=float, default=0.0) # 0.9

    parser.add_argument("--comment", type=str, default="No comment")

    args = parser.parse_args()
    print(args.load_model)
    # ======================================================================
    # load options if needed
    # ======================================================================

    if args.load_options != '':
        print("Load options form:", args.load_options)
        skipkeys = ['size_batch', 'length_epoch', 'num_data_fake' ,'device_cuda', 'validation_freq', 'comment'
                'print_freq', 'compute_fid', 'load_model', 'load_options', 'fake_data_warmup_at', 'fake_data_warmup_every', 
                'length_epoch', 'scheduler_arg_step', 'scheduler_arg_1', 'scheduler_arg_2', 'scheduler_type', 'fake_train_jumps', 'refresh_rate']
        # Load args
        f = open(args.load_options)
        dict_obj = json.load(f)
        f.close()

        # MAKE args
        parser = argparse.ArgumentParser()
        for key, value in dict_obj.items():
            if key in skipkeys:
                continue
            setattr(args, key, value)
            # parser.add_argument(f"--{key}", default=value)
        # args = parser.parse_args()

    if args.noise_end == 0.0:
        args.noisy_real = False
    
    return args


def get_dataset(args):

    # ======================================================================
    # dataset 
    # ======================================================================
    # dir_data = os.path.join('/ssd1', 'dataset')
    # /ssd1/dataset/metfaces/metfaces_mini/

    if args.name_dataset.upper() == 'MNIST':
        args.channel_data = 1
        transform = torchvision.transforms.Compose([ 
            torchvision.transforms.Resize([args.height_data, args.width_data]),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Lambda(lambda t: (t - torch.mean(t)) / torch.std(t)) # mean 0, std 1
            torchvision.transforms.Lambda(lambda t: 2.0 * t - 1) 
        ])
        dataset = torchvision.datasets.MNIST(dir_data, transform=transform, train=True, download=True)
    elif args.name_dataset.upper() == 'CIFAR10':
        dir_data = '../data/cifar10_folder'
        args.channel_data = 3
        args.height_data = 32
        args.width_data = 32

        transform=torchvision.transforms.Compose([
            # torchvision.transforms.Resize(args.height_data, antialias=False),
            # torchvision.transforms.CenterCrop(args.height_data),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Lambda(lambda t: 2.0 * t - 1) 
        ])
        dataset = torchvision.datasets.ImageFolder(root=dir_data, transform=transform)

    elif args.name_dataset.upper() == 'CELEBA64':
        dir_data = '../data/celeba64/'
        args.channel_data = 3
        args.height_data = 64
        args.width_data = 64

        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.height_data, antialias=False),
            torchvision.transforms.CenterCrop(args.height_data),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Lambda(lambda t: 2.0 * t - 1) 
        ])
        dataset = torchvision.datasets.ImageFolder(root=dir_data, transform=transform)
    
    elif args.name_dataset.upper() == 'CELEBAHQ64':
        dir_data = '../data/celebahq64'
        args.channel_data = 3
        args.height_data = 64
        args.width_data = 64

        transform=torchvision.transforms.Compose([
            # torchvision.transforms.Resize(args.height_data, antialias=False),
            # torchvision.transforms.CenterCrop(args.height_data),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Lambda(lambda t: 2.0 * t - 1) 
        ])
        dataset = torchvision.datasets.ImageFolder(root=dir_data, transform=transform)
    
    elif args.name_dataset.upper() == 'AFHQV264':
        dir_data = '../data/afhqv264/'
        args.channel_data = 3
        args.height_data = 64
        args.width_data = 64

        transform=torchvision.transforms.Compose([
            # torchvision.transforms.Resize(args.height_data, antialias=False),
            # torchvision.transforms.CenterCrop(args.height_data),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Lambda(lambda t: 2.0 * t - 1) 
        ])
        dataset = torchvision.datasets.ImageFolder(root=dir_data, transform=transform)
    
    elif args.name_dataset.upper() == 'LSUNCONFERENCE64':
        dir_data = '../data/lsun_conference_room64/'
        args.channel_data = 3
        args.height_data = 64
        args.width_data = 64

        transform=torchvision.transforms.Compose([
            # torchvision.transforms.Resize(args.height_data, antialias=False),
            # torchvision.transforms.CenterCrop(args.height_data),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Lambda(lambda t: 2.0 * t - 1) 
        ])
        dataset = torchvision.datasets.ImageFolder(root=dir_data, transform=transform)
    elif args.name_dataset.upper() == 'LSUNCHURCH64':
        dir_data = '../data/lsun_church_outdoor64/'
        args.channel_data = 3
        args.height_data = 64
        args.width_data = 64

        transform=torchvision.transforms.Compose([
            # torchvision.transforms.Resize(args.height_data, antialias=False),
            # torchvision.transforms.CenterCrop(args.height_data),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Lambda(lambda t: 2.0 * t - 1) 
        ])
        dataset = torchvision.datasets.ImageFolder(root=dir_data, transform=transform)
    
    if args.num_data_real == -1:
        args.num_data_real   = len(dataset)
    if args.num_data_fake == 0:
        args.num_data_fake = args.num_data_real

    if args.name_dataset.upper() == 'MNIST' or args.name_dataset.upper() == 'CIFAR10':
        if args.use_subset:

            use_label       = args.use_subset_label
            idx_label       = (dataset.targets == use_label)
            dataset.data    = dataset.data[idx_label]
            
            dataset.targets = dataset.targets[idx_label]
        dataset.data    = dataset.data[0:args.num_data_real]
        dataset.targets = dataset.targets[0:args.num_data_real]
    else:
        # print(dataset.samples[0:args.num_data_real])
        dataset.samples    = dataset.samples[0:args.num_data_real]

    print("Number of data:", len(dataset))
    if args.in_memory_dataset:
        dataset = InMemoryDatasetReal(args, dataset)
    return dataset



def get_model(args):
    # ======================================================================
    # model and optimizer 
    # ======================================================================



    #from models.discriminator import Discriminator
    if args.name_model.lower() == 'resnet':
        import models.igebm.model_time3 as igebm 
        model = igebm.get_ebm_model()

    elif args.name_model.lower() == 'unet_improved2_attpool2_cone': #
        import models.resnet_based.improved_wrapper as improved_wrapper 
        model = improved_wrapper.create_model_wrapper_cone(image_size=args.height_data, length_time=args.length_time)
    elif args.name_model.lower() == 'unet_improved2_attpool2_small': 
        import models.unet_based.improved_wrapper as improved_wrapper 
        model = improved_wrapper.create_model_wrapper_small2(image_size=args.height_data, length_time=args.length_time)

        
    
    


    # LOAD MODEL WEIGHT IF NEEDED
    if args.load_model != '':
        print("Load model form:", args.load_model)
        device = torch.device(f'cuda:{args.device_cuda}' if torch.cuda.is_available() else 'mps')

        load_dict = torch.load(args.load_model, map_location=device)
        model.load_state_dict(load_dict['model_state_dict'], strict=True) 
    
    print("Parameter count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model



def get_optimizer(args, model):
    if args.option_optim.lower() == 'sgd':
        optim_energy = torch.optim.SGD(model.parameters(), lr=args.lr_energy)
    elif args.option_optim.lower() == 'adam':
        optim_energy = torch.optim.Adam(model.parameters(), lr=args.lr_energy, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)
    elif args.option_optim.lower() == 'adamw':
        optim_energy = torch.optim.AdamW(model.parameters(), lr=args.lr_energy, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)#, betas=(0.5,0.9))
    elif args.option_optim.lower() == 'adamsn':
        optim_energy = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_energy, betas=(0.0,0.9))
    elif args.option_optim.lower() == 'adamwsn':
        optim_energy = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_energy, betas=(0.0,0.9))
    elif args.option_optim.lower() == "sam":
        base_optimizer = torch.optim.Adam
        args.rho = 2
        optim_energy = SAM(model.parameters(), base_optimizer, lr=args.lr_energy, betas=[.9, .999], weight_decay=args.weight_decay, rho=args.rho, adaptive=False if args.rho < 0.5 else True)
        print ("USING SAM OPTIMIZER")
    if args.load_model != '':
        print("Load optimizer form:", args.load_model)
        device = torch.device(f'cuda:{args.device_cuda}' if torch.cuda.is_available() else 'mps')

        load_dict = torch.load(args.load_model) # , map_location=device
        optim_energy.load_state_dict(load_dict['optimizer_state_dict']) 
    
    return optim_energy




class CosineAnnealingWarmRestartsDecay(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1,
                    eta_min=0, last_epoch=-1, verbose=False, decay=1):
        super().__init__(optimizer, T_0, T_mult=T_mult,
                            eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
        self.decay = decay
        self.initial_lrs = self.base_lrs
    
    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0
            
            self.base_lrs = [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)

def get_scheduler(args, optim_energy):
    if args.scheduler_type.lower() == 'lambda':
        milestone = 500*args.model_update_num
        decay_steps = args.length_epoch*args.model_update_num - milestone
        lambda_lr = lambda step:  1.0 if step <= milestone else 0.001**((step-milestone+1)/decay_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda=lambda_lr)
    elif args.scheduler_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=args.scheduler_arg_step*args.model_update_num, gamma=args.scheduler_gamma)
    elif args.scheduler_type.lower() == 'multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_energy, milestones=[500,1000,1500,2000,2500], gamma=args.scheduler_gamma)
    elif args.scheduler_type.lower() == 'linear':
        decay_steps = args.length_epoch*args.model_update_num
        lambda_lr = lambda step:  1.0 - step/decay_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda=lambda_lr)
    elif args.scheduler_type.lower() == 'exp':
        total_multilipier = args.scheduler_arg_1/args.lr_energy

        gamma = total_multilipier**(1/(args.length_epoch*args.model_update_num))
        print (total_multilipier)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_energy, gamma)
    elif args.scheduler_type.lower() == 'cyclic':
        # num_batch   = len(dataloader)
        lr_max = args.scheduler_arg_1
        lr_min = args.scheduler_arg_2
        ratio       = (lr_min - args.lr_energy) / (lr_max - args.lr_energy)
        log_decay   = np.log(ratio) / (args.length_epoch*args.model_update_num)
        decay       = np.exp(log_decay)
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optim_energy,
            base_lr=args.lr_energy,
            max_lr=lr_max,
            step_size_up= args.scheduler_arg_step,
            mode="exp_range",
            gamma=decay,
            cycle_momentum=False,   # this must be falst with AdamW
            # verbose=True,
            )
    elif args.scheduler_type.lower() == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optim_energy, step_size=args.scheduler_arg_step*args.model_update_num, gamma=1)
    elif args.scheduler_type.lower() == 'dropoff':
        milestone = args.scheduler_arg_step*args.model_update_num
        decay_steps = (args.length_epoch - args.scheduler_arg_step)*args.model_update_num
        lambda_lr = lambda step:  1.0 if step <= milestone else 1.0 - (step-milestone+1)/decay_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda=lambda_lr)
    elif args.scheduler_type.lower() == 'dropoffexp':
        milestone = args.scheduler_arg_step*args.model_update_num
        decay_steps = (args.length_epoch - args.scheduler_arg_step)*args.model_update_num
        lambda_lr = lambda step:  1.0 if step <= milestone else args.scheduler_arg_1**((step-milestone+1)/decay_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda=lambda_lr)
    elif args.scheduler_type.lower() == 'cosineannealingwarmrestarts':
        warmup = args.scheduler_arg_step*args.model_update_num+1
        scheduler_const = torch.optim.lr_scheduler.LinearLR(optim_energy, start_factor=1.0/warmup, total_iters=warmup)

        scheduler_anneal = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim_energy,
            T_0= int(args.scheduler_arg_1*args.model_update_num),
            T_mult=1,
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(optim_energy, schedulers=[scheduler_const, scheduler_anneal], milestones=[warmup])
    elif args.scheduler_type.lower()== 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optim_energy,
            T_max=args.length_epoch*args.model_update_num,
            )
            
    elif args.scheduler_type.lower() == 'cosineannealingwarmrestartsdecay':
        warmup = args.scheduler_arg_step*args.model_update_num+1
        scheduler_const = torch.optim.lr_scheduler.LinearLR(optim_energy, start_factor=1.0/warmup, total_iters=warmup)

        total_steps = (args.length_epoch*args.model_update_num-warmup)//(args.scheduler_arg_1*args.model_update_num)
        decay = (args.scheduler_arg_2/args.lr_energy)**(1/total_steps)

        scheduler_anneal = CosineAnnealingWarmRestartsDecay(
            optim_energy,
            T_0= int(args.scheduler_arg_1*args.model_update_num),
            T_mult=1,
            decay=decay
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(optim_energy, schedulers=[scheduler_const, scheduler_anneal], milestones=[warmup])
    
    elif args.scheduler_type.lower() == 'warmupconst':
        warmup = args.scheduler_arg_step*args.model_update_num+1
        scheduler_const = torch.optim.lr_scheduler.LinearLR(optim_energy, start_factor=1.0/warmup, total_iters=warmup)
        lambda_lr = lambda step:  1.0 
        scheduler_anneal = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda=lambda_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optim_energy, schedulers=[scheduler_const, scheduler_anneal], milestones=[warmup])
    
    elif args.scheduler_type.lower() == 'warmupdropoffexp':
        warmup = args.scheduler_arg_2*args.model_update_num+1
        scheduler_const = torch.optim.lr_scheduler.LinearLR(optim_energy, start_factor=1.0/warmup, total_iters=warmup)

        milestone = args.scheduler_arg_step*args.model_update_num - warmup
        decay_steps = (args.length_epoch - args.scheduler_arg_step)*args.model_update_num
        lambda_lr = lambda step:  1.0 if step <= milestone else args.scheduler_arg_1**((step-milestone+1)/decay_steps)
        scheduler_anneal = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda=lambda_lr)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optim_energy, schedulers=[scheduler_const, scheduler_anneal], milestones=[warmup])
    
    elif args.scheduler_type.lower() == 'warmupdropofflinear':
        warmup = args.scheduler_arg_2*args.model_update_num+1
        scheduler_const = torch.optim.lr_scheduler.LinearLR(optim_energy, start_factor=1.0/warmup, total_iters=warmup)

        milestone = args.scheduler_arg_step*args.model_update_num - warmup
        decay_steps = (args.length_epoch - args.scheduler_arg_step)*args.model_update_num
        lambda_lr = lambda step:  1.0 if step <= milestone else 1.0 - (step-milestone+1)/decay_steps
        scheduler_anneal = torch.optim.lr_scheduler.LambdaLR(optim_energy, lr_lambda=lambda_lr)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optim_energy, schedulers=[scheduler_const, scheduler_anneal], milestones=[warmup])
    # if args.load_model != '':
    #     print("Load scheduler form:", args.load_model)
    #     device = torch.device(f'cuda:{args.device_cuda}' if torch.cuda.is_available() else 'mps')

    #     load_dict = torch.load(args.load_model, map_location=device)
    #     optim_energy.load_state_dict(load_dict['scheduler_state_dict']) 
    return scheduler




# =============================================================================
# fake dataset
# =============================================================================


class DatasetFake(Dataset):
    def __init__(self, args):
        self.number_data    = args.num_data_fake
        self.length_time    = args.length_time 
        # self.data   = torch.randn(args.num_data_fake, args.channel_data, args.height_data, args.width_data)
        # print(self.data.size())
        if args.batch_type == "mixed_random":
            self.time        = torch.randint(0, args.length_time, (args.num_data_fake,))
        elif args.batch_type == "mixed_random_jumps":
            self.time        = torch.randint(0, args.length_time//args.fake_train_jumps, (args.num_data_fake,))
            self.time = self.time * args.fake_train_jumps
        else:
            self.time    = torch.zeros(args.num_data_fake).type(torch.LongTensor)
        self.label       = torch.randint(0, 9, (args.num_data_fake, 1))
        
        if args.init_from == "gaussian":
            self.init_random = self._init_gaussian
        elif args.init_from == "informed": 
            self._init_from_centers(args)
            self.init_random = self._init_informed

        if args.buffer_type == "normal":
            self.getitem = self._getitem_normal
        elif args.buffer_type == "replay_buffer": 
            self.getitem = self._getitem_replay
            self.refresh_rate = args.refresh_rate
        elif args.buffer_type == "replay_buffer_time": 
            self.getitem = self._getitem_replay_time
            self.refresh_rate = args.refresh_rate

        self.args   = args
        self.data   = self.init_random(args, args.num_data_fake, self.label)
        print(self.data.size())

    def __len__(self):
        return self.number_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.getitem(idx)

    def _getitem_normal(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.data[idx]
        time = self.time[idx]
        label = self.label[idx]
        return data, time, label, idx


    def _getitem_replay(self, idx):
        if isinstance(idx, int):
            bs = 1
        else:
            bs = len(idx)
        data = self.data[idx]
        time = self.time[idx]
        label = self.label[idx]

        
        # print(random_samples.size(), random_samples[0].size())
        if torch.rand(bs).item() < self.refresh_rate:
            # random_samples = self.init_random(self.args, bs, label)[0]
            # data = choose_random * random_samples + (1 - choose_random) * data
            time *= 0
            data = self.init_random(self.args, bs, label)[0]
        return data, time, label, idx

    def _getitem_replay_time(self, idx):
        if isinstance(idx, int):
            bs = 1
        else:
            bs = len(idx)
        data = self.data[idx]
        time = self.time[idx]
        label = self.label[idx]

        if time == self.args.length_time-1:
            if torch.rand(bs).item() < self.refresh_rate:
                time *= 0
                data = self.init_random(self.args, bs, label)[0]
        return data, time, label, idx

    def _update_data(self, data: torch.Tensor, idx):
        self.data[idx] = data

    def _update_time(self, time: torch.Tensor, idx):
        self.time[idx] = time

    def _update_label(self, label: torch.Tensor, idx):
        self.label[idx] = label

    def _update(self, data: torch.Tensor, time: torch.Tensor, label: torch.Tensor, idx):
        self.data[idx] = data
        self.time[idx] = time
        self.label[idx] = label

    def _init_gaussian(self, args, bs, labels):
        return torch.randn(bs, args.channel_data, args.height_data, args.width_data)

    def _init_informed(self, args, bs, labels):

        size = [bs, args.channel_data, args.height_data, args.width_data]
        new = torch.zeros(bs, size[1], size[2], size[3])
        # for i in range(bs):
        dist = self.conditionals[0]#[label]
        new = dist.sample_n(bs).view(size)
        # return torch.clamp(new, -1, 1)
        # print("INIT SHAPE", new.size())
        return new

    def _init_from_centers(self, args):
        from torch.distributions.multivariate_normal import MultivariateNormal
        
        size = [args.channel_data, args.height_data, args.width_data]

        centers = torch.load('%s/%s_mean.pt' % ('../data/mean_inits', args.name_dataset))
        covs = torch.load('%s/%s_cov.pt' % ('../data/mean_inits', args.name_dataset))

        self.conditionals = []
        for i in range(1): # args.n_classes
            mean = centers[i]#.to(args.device)
            cov = covs[i]#.to(args.device)
            dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * torch.eye(int(np.prod(size))) )#.to(args.device))

            self.conditionals.append(dist)
        

def get_dataset_fake(args):
    # if args.num_data_fake == 0:
    #     args.num_data_fake = (args.num_data_real // args.batch_size) * args.batch_size 
    dataset = DatasetFake(args)
    return dataset

class InMemoryDatasetReal(Dataset):
    def __init__(self, args, dataset):
        self.number_data    = args.num_data_real
       
        self.data   = torch.randn(args.num_data_real, args.channel_data, args.height_data, args.width_data)
        self.label       = torch.randint(0, 9, (self.number_data, 1))

        test_loader = DataLoader(dataset=dataset, batch_size=1, drop_last=False, shuffle=False)

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader, 0):
                self.data[i] = images[0]
                self.label[i] = self.label[0]
                print (i+1,"out of", args.num_data_real ,"files completed", end='\r')
        

    def __len__(self):
        return self.number_data

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label

    

if __name__ == '__main__': 
    dir_data = '/hdd1/dataset/metfaces/'
    channel_data = 3
    height_data = 64
    width_data = 64

    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(height_data, antialias=False),
        torchvision.transforms.CenterCrop(height_data),
        torchvision.transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # torchvision.transforms.Lambda(lambda t: 2.0 * t - 1) 
    ])
    dataset = torchvision.datasets.ImageFolder(root=dir_data, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, drop_last=False, shuffle=False)
    import os
    
    from PIL import Image
    f = open("test_y", "w")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader, 0):
            # outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            
            sample_fname, _ = test_loader.dataset.samples[i]
            print(images.size())
            print(os.path.basename(sample_fname))
            image = torchvision.transforms.functional.to_pil_image(images[0])
            image.save('../data/metfaces_mini/'+os.path.basename(sample_fname), format='PNG')

    f.close()


