from torch.nn.parameter import Parameter
import torchvision
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import pytorch_lightning as pl
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from losses import compute_loss_positive, compute_total_variation, compute_loss_negative, compute_gradient_penalty

from models.simple.energy_model import Energy
from models.diffusiongan.discriminator import get_diffusion_discriminator
#from models.discriminator import Discriminator
from inference.figures import save_inference, save_50, save_all_time, compute_fid, save_similarities, save_similarities_new, save_50_losses

from diffusion import NoiseSampler
from trainer import Trainer

from torch.autograd import Variable
import math

def inference(model, fake, label, args, trainer=None):
    device = torch.device(f'cuda:{args.device_cuda}' if torch.cuda.is_available() else 'mps')
    old_fake_train_jumps = args.fake_train_jumps
    old_random_jumps = args.random_jumps
    args.fake_train_jumps = 1
    args.random_jumps = False
    if trainer is None:
        trainer = Trainer(model, None, None, args, None, None, None, device)
    trainer.num_batch_fake      = 1
    time        = torch.zeros(args.size_batch).type(torch.IntTensor).to(device)
    fake        = fake.clone().to(device)
    label       = label.to(device)
    iteration_data = []
    for i in range(args.length_time//args.fake_train_jumps): 
        fake = trainer.update_fake(fake, time, label, args)
        iteration_data.append(fake.detach().cpu().clone().numpy())
        time += args.fake_train_jumps
        
        print ("INFERENCE:",i+1,"out of", args.length_time//args.fake_train_jumps ,"time steps complete", end='\r')
    print('')
    args.fake_train_jumps = old_fake_train_jumps
    args.random_jumps = old_random_jumps
    return iteration_data, label.detach().clone().cpu().squeeze().numpy()


def inference_train(model, fake, label_fake, args, trainer):
    device = torch.device(f'cuda:{args.device_cuda}' if torch.cuda.is_available() else 'mps')

    time        = torch.zeros(args.size_batch).type(torch.IntTensor).to(device)
    fake        = fake.clone().to(device)
    label       = label_fake.to(device)
    iteration_data = []

    args.lr_energy = 0
    for i in range(args.length_time): 
        fake = trainer.update_fake(fake, time, label, args)
        real, fake, time = trainer.update_model(fake, time, label, args)
        # trainer.update_fake(idx_dataset_fake, args)

        iteration_data.append(fake.detach().cpu().clone().numpy())
        time += 1
        print ("INFERENCE:",i+1,"out of", args.length_time ,"time steps complete", end='\r')
    print('')

    return iteration_data, label.detach().clone().cpu().squeeze().numpy()

def inference_with_real(model, fake, label_fake, args, trainer=None):
    device = torch.device(f'cuda:{args.device_cuda}' if torch.cuda.is_available() else 'mps')
    if trainer is None:
        trainer = Trainer(model, None, None, args, None, None, None, device)
    trainer.num_batch_fake      = 1
    time        = torch.zeros(args.size_batch).type(torch.IntTensor).to(device)
    fake        = fake.clone().to(device)
    label       = label_fake.to(device)
    iteration_data = []
    for i in range(args.length_time//args.fake_train_jumps):
        try:
            real, label_real = next(trainer.data_iter) 
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader 
            trainer.data_iter = iter(trainer.dataloader)
            real, label_real = next(trainer.data_iter) 
        real = real.to(device)
        label_real = label_real.to(device).unsqueeze(1)
        pred_positive       = trainer.model(real, time.to(device), label_real)    
        trainer.val_pred_real.append(pred_positive.mean().item())

        fake = trainer.update_fake(fake, time, label, args)
        # fake = torch.clip(fake, -1, 1)
        iteration_data.append(fake.detach().cpu().clone().numpy())
        time += args.fake_train_jumps
        
        print ("INFERENCE:",i+1,"out of", args.length_time ,"time steps complete", end='\r')
    print('')
    return iteration_data, label.detach().clone().cpu().squeeze().numpy()

def cosine_similarity(fake, real):
    fake = torch.from_numpy(fake)
    fake = fake.flatten(start_dim=1)
    real = real.flatten(start_dim=1)

    cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-08)
    output = cos(real[None, :, :], fake[:, None, :])
    
    print(output.size())
    return output

def cosine_similarity_new(fake, tmp_dataloader):
    cos = torch.nn.CosineSimilarity(dim=-1,eps=1e-08)
    fake = torch.from_numpy(fake)
    fake = fake.flatten(start_dim=1)
    # sample the entire dataset of real
    
    original_real=[]
    for i, (imgs, _) in enumerate(tmp_dataloader):
        raw_real = Variable(imgs).type(torch.FloatTensor)
        real = raw_real.flatten(start_dim=1)
        similarities = cos(real[None, :, :], fake[:, None, :])
        if i >0 :
            temp_max, temp_indx = torch.max(similarities, 1)
            change = (temp_max > maxes).nonzero().squeeze()
            # print(change)
            maxes[change] = temp_max[change]
            # print(change.size())
            sim_reals[change] = raw_real[temp_indx[change]]
        else:
            maxes, mindx = torch.max(similarities, 1)
            sim_reals = raw_real[mindx]
        print ("SIMLIARITY:",i+1,"out of", len(tmp_dataloader) ,"batches complete", end='\r')
    print('')
    print(sim_reals.size())
    return maxes, sim_reals
