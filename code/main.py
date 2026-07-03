import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn

import pytorch_lightning as pl

from logger import Logger
from trainer import Trainer

from arguments import get_args, get_dataset, get_model, get_optimizer, get_scheduler, get_dataset_fake


if __name__ == '__main__': 
    # ======================================================================
    # random seed
    # ======================================================================
    pl.seed_everything(0)

    #======================================================================
    # get command line arguments 
    # ======================================================================
    args = get_args()
    args.world_size = 1

    # ======================================================================
    # device
    # ======================================================================
    device = torch.device(f'cuda:{args.device_cuda}' if torch.cuda.is_available() else 'mps')

    # ======================================================================
    # dataset 
    # ======================================================================
    dataset = get_dataset(args)
    dataset_fake = get_dataset_fake(args)
    # ======================================================================
    # model 
    # ======================================================================

    model = get_model(args).to(device)
    # mock DDP

    # ======================================================================
    # optimizer and scheduler
    # ======================================================================
    optim_energy = get_optimizer(args, model)

    scheduler = get_scheduler(args, optim_energy)

    # ======================================================================
    # Logger
    # ======================================================================
    logger = Logger(model, dataset, args, fake_data_init_func=dataset_fake.init_random)

    # ======================================================================
    # training 
    # ======================================================================

    model.train()

    trainer = Trainer(model, dataset, dataset_fake, args, logger, scheduler, optim_energy, device)
    trainer.start_train()

