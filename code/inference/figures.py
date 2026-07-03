from torch.nn.parameter import Parameter
import torch
import matplotlib.pyplot as plt
import numpy as np
from losses import compute_total_variation, compute_loss_negative
import json
import argparse
import pytorch_lightning as pl

from models.simple.energy_model import Energy
from models.diffusiongan.discriminator import get_diffusion_discriminator

from torch.autograd import Variable
import inference.fid_score as fid # FID metric
#from models.discriminator import Discriminator
import os
import torchvision
from PIL import Image
from tqdm import tqdm

def compute_fid_torchivison(tmp_dataloader, fake_batch, batch_size, sample_size=0):
    if sample_size == 0:
        sample_size = batch_size
    #tmp_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=2)
    with torch.no_grad():
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=2048)

        original_real=[]
        original_fake=fake_batch
        for i, (imgs, _) in enumerate(tmp_dataloader):
                original_real.append(Variable(imgs).type(torch.FloatTensor))

                # we sameple a subset of the examples
                if (i+1) * batch_size >= sample_size:
                    break
        print(len(original_fake), len(original_real))
        original_real = torch.cat(original_real,dim=0)
        original_fake = torch.cat(original_fake,dim=0)


        fid = compute_activations(fid, original_real, batch_size, True)
        fid = compute_activations(fid, original_fake, batch_size, False)

        fid_original = float(fid.compute())
        print("fid", fid_original)
    return fid_original

def compute_activations(fid, batch, batch_size, real=True):
    for i in tqdm(range(0, batch.shape[0], batch_size)): #tqdm: progressive bar

        start = i
        end = i + batch_size

        data = batch[start:end]

        fid.update(data.type(torch.uint8), real=real)

    return fid

def compute_fid(tmp_dataloader, fake_batch, batch_size, sample_size=0):
    if sample_size == 0:
        sample_size = batch_size
    #tmp_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=2)
    with torch.no_grad():
        
        original_real=[]
        original_fake=fake_batch
        for i, (imgs, _) in enumerate(tmp_dataloader):
                original_real.append(Variable(imgs).type(torch.FloatTensor))

                # we sameple a subset of the examples
                if (i+1) * batch_size >= sample_size:
                    break
        original_real = torch.cat(original_real,dim=0)
        original_fake = torch.cat(original_fake,dim=0)

        
        fid_original = fid.calculate_fid_given_batches(original_real, original_fake, batch_size=batch_size)
    print("fid", fid_original)
    return fid_original

def save_inference(iteration_data, save_path, label_fake, args):
    # -------------------------------------------------------------------
    # save the figure
    # -------------------------------------------------------------------     
    time_len = iteration_data.shape[0] #args.length_time//args.fake_train_jumps  
    if time_len == 1:
        return
    nRow    = 5
    nCol    = time_len
    colIndex = np.arange(0, 10)
    if nCol > 10:
        if time_len%10 == 0:
            nCol = 11
            colIndex = np.arange(0, time_len, time_len//10)
            colIndex = np.append(colIndex, [time_len-1])
        else:
            nCol = time_len//(time_len//10+1)
            colIndex = np.arange(0, time_len, time_len//10+1)
            colIndex = np.append(colIndex, [time_len-1])

    fSize   = 3
    # print(colIndex)
    fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

    plt_cmap = 'gray'
    if args.channel_data == 3:
        iteration_data = np.moveaxis(iteration_data, 2, -1)
        iteration_data = (iteration_data+1)/2
        plt_cmap = 'viridis'

    for c in range(nCol):
        for r in range(nRow):
            ax[r][c].set_title('time = ' + str(colIndex[c]) + ' c = ' + str(label_fake[r]))
            ax[r][c].imshow(iteration_data[colIndex[c]][r].squeeze(), cmap=plt_cmap)

        
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def save_50(iteration_data, save_path, label_fake, args, sample_t=1):
    # -------------------------------------------------------------------
    # save the figure
    # -------------------------------------------------------------------         
    nRow    = 5
    nCol    = 10
    fSize   = 3
    max_time   = iteration_data.shape[0] #args.length_time//args.fake_train_jumps  

    fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

    plt_cmap = 'gray'
    
    if args.channel_data == 3:
        iteration_data = np.moveaxis(iteration_data, 2, -1)
        iteration_data = (iteration_data+1)/2
        plt_cmap = 'viridis'

    for c in range(nCol):
        for r in range(nRow):
            if r+nRow*c >= len(label_fake):
                break
            ax[r][c].set_title('time = ' + str(max_time-sample_t) + ' c = ' + str(label_fake[r+nRow*c]))
            ax[r][c].imshow(iteration_data[max_time-sample_t][r+nRow*c].squeeze(), cmap=plt_cmap)

        
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def save_50_losses(iteration_data, save_path, label_fake, args, sample_t=1, values=0):
    # -------------------------------------------------------------------
    # save the figure
    # -------------------------------------------------------------------         
    nRow    = 5
    nCol    = 10
    fSize   = 3
    max_time   = iteration_data.shape[0] #args.length_time//args.fake_train_jumps  

    fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

    plt_cmap = 'gray'
    
    if args.channel_data == 3:
        iteration_data = np.moveaxis(iteration_data, 2, -1)
        iteration_data = (iteration_data+1)/2
        plt_cmap = 'viridis'

    for c in range(nCol):
        for r in range(nRow):
            if r+nRow*c >= len(label_fake):
                break
            ax[r][c].set_title('loss = ' + str(values[[r+nRow*c]].item()))
            ax[r][c].imshow(iteration_data[max_time-sample_t][r+nRow*c].squeeze(), cmap=plt_cmap)

        
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def save_spike_train(train_batch, save_path, args, alphas):
    # -------------------------------------------------------------------
    # save the figure
    # -------------------------------------------------------------------         
    nRow    = 5
    nCol    = 10
    fSize   = 3

    fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

    plt_cmap = 'gray'
    
    if args.channel_data == 3:  
        train_batch = np.clip(np.moveaxis(train_batch, 1, -1), -1, 1)
        train_batch = (train_batch+1)/2
        plt_cmap = 'viridis'

    for c in range(nCol):
        for r in range(nRow):
            if r+nRow*c >= len(alphas):
                break
            ax[r][c].set_title('alpha = ' + str(alphas[r+nRow*c].item()))
            ax[r][c].imshow(train_batch[r+nRow*c].squeeze(), cmap=plt_cmap)

        
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def save_all_time(iteration_data, save_path, label_fake, args):
    # -------------------------------------------------------------------
    # save the figure
    # -------------------------------------------------------------------         
    nRow    = 5
    nCol    = 10
    fSize   = 3
    max_time   = args.length_time

    plt_cmap = 'gray'
    if args.channel_data == 3:
        
        iteration_data = ((iteration_data.permute(0, 2, 3, 1)+1)/2*255).numpy().astype(np.uint8)
        plt_cmap = 'viridis'

    fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))
    time_it = 0
    for c in range(nCol):
        for r in range(nRow):

            ax[r][c].set_title('time = ' + str(time_it) + ' c = ' + str(label_fake[0]))
            ax[r][c].imshow(iteration_data[time_it][0].squeeze(), cmap=plt_cmap)
            time_it += 1

        
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

def save_similarities(fakes, reals, similarities, save_path, label_fake, args):
    # -------------------------------------------------------------------
    # save the figure
    # -------------------------------------------------------------------         
    nRow    = 5
    nCol    = 10
    fSize   = 3
    max_time   = args.length_time

    fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

    plt_cmap = 'gray'
    
    if args.channel_data == 3:
        fakes = np.moveaxis(fakes, 1, -1)
        fakes = (fakes+1)/2

        reals = np.moveaxis(reals, 1, -1)
        reals = (reals+1)/2
        plt_cmap = 'viridis'

    for r in range(nRow):
        for c in range(0, nCol, 2):
            idx = int((c+r*nCol)/2)
            if idx >= len(label_fake):
                break
            
            ax[r][c].set_title('time = ' + str(max_time-1) + ' c = ' + str(label_fake[idx]))
            ax[r][c].imshow(fakes[idx].squeeze(), cmap=plt_cmap)
            # print(similarities[idx].size(), similarities.size())
            sidx = np.argmax(similarities[idx])
            ax[r][c+1].set_title('similarity = ' + str(similarities[idx, sidx]))
            ax[r][c+1].imshow(reals[sidx].squeeze(), cmap=plt_cmap)
        
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def save_similarities_new(fakes, reals, similarities, save_path, label_fake, args):
    # -------------------------------------------------------------------
    # save the figure
    # -------------------------------------------------------------------         
    nRow    = 5
    nCol    = 10
    fSize   = 3
    max_time   = args.length_time

    fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

    plt_cmap = 'gray'
    
    if args.channel_data == 3:
        fakes = np.moveaxis(fakes, 1, -1)
        fakes = (fakes+1)/2

        reals = np.moveaxis(reals, 1, -1)
        reals = (reals+1)/2
        plt_cmap = 'viridis'

    for r in range(nRow):
        for c in range(0, nCol, 2):
            idx = int((c+r*nCol)/2)
            if idx >= len(label_fake):
                break
            
            ax[r][c].set_title('time = ' + str(max_time-1) + ' c = ' + str(label_fake[idx]))
            ax[r][c].imshow(fakes[idx].squeeze(), cmap=plt_cmap)
            ax[r][c+1].set_title('similarity = ' + str(similarities[idx]))
            ax[r][c+1].imshow(reals[idx].squeeze(), cmap=plt_cmap)
        
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)



