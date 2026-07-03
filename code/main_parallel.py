import torch
import torchvision
from torch.utils.data import Dataset
from os.path import join
from torch.utils.data import DataLoader

import datetime
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

from arguments import get_args, get_dataset, get_model, get_optimizer, get_scheduler, get_dataset_fake

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        # num_workers=0,
        drop_last=True
    )

def thread_main(rank, world_size, args, dataset_fake):
    # ======================================================================
    # device
    # ======================================================================
    ddp_setup(rank, world_size)
    
    # args.num_data_fake = args.num_data_fake//world_size
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'mps')
    print(device)

    dataset = get_dataset(args)
    # dataset_fake = get_dataset_fake(args)
    model = get_model(args).to(device)

    # params = model.named_parameters() 
    # i = 0
    # for name, param in params:
    #     if i in [44, 45, 64, 65, 80, 81]:
    #         print(name, i)
    #     i += 1
    # layers = model.named_modules() #list(model.parameters())
    # print(layers[44].shape, layers[45].shape)
    optim_energy = get_optimizer(args, model)
    scheduler = get_scheduler(args, optim_energy)
    
    if rank == 0:
        logger = Logger(model, dataset, args, dataloader_func=prepare_dataloader, fake_data_init_func=dataset_fake.init_random)
    else:
        logger = None
    # print("print(STARTING TRAINIGN LOOP)")
    model = DDP(model, device_ids=[rank])
    # print("print(STARTING TRAINIGN LfasfasfOOP)")
    trainer = Trainer(model, dataset, dataset_fake, args, logger, scheduler, optim_energy, device, dataloader_func=prepare_dataloader)
    try:
        trainer.start_train()
        destroy_process_group()
    except BaseException as e:
        destroy_process_group()
        raise e
    # destroy_process_group()

    # train_data = prepare_dataloader(dataset, batch_size)
    # trainer = Trainer(model, train_data, optimizer, rank, save_every)
    # trainer.train(total_epochs)
    
if __name__ == '__main__': 
    # ======================================================================
    # random seed
    # ======================================================================
    pl.seed_everything(0)

    #======================================================================
    # get command line arguments 
    # ======================================================================
    args = get_args()

    world_size = torch.cuda.device_count()
    args.size_batch = args.size_batch//world_size
    args.world_size = world_size

    # args.lr_fake = args.lr_fake/4 * args.world_size args.size_batch/32
    dataset = get_dataset(args) # UNSED BUT NEEDED TO UPDATE ARGS
    
    dataset_fake = get_dataset_fake(args)
    
    try:
        print("Running parallel scripts.")
        mp.spawn(thread_main, args=(world_size, args, dataset_fake), nprocs=world_size)
    except BaseException as e:
        print(e)
        if str(e) !="":
        
            now         = datetime.datetime.now()
            date_stamp  = now.strftime('%Y_%m_%d') 
            time_stamp  = now.strftime('%H_%M_%S')
            print("error{}.log".format(time_stamp))
            logf = open("error{}.log".format(time_stamp), "w")
            logf.write(str(e))
            logf.close()
        # destroy_process_group() 
        # try: 
        #     destroy_process_group()  
        # except: 
        #     print("failed destroy")
        #     os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")

        


