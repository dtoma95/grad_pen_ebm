import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from torch.nn.parameter import Parameter
import torchvision
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

from models.diffusiongan.discriminator import get_diffusion_discriminator
#from models.discriminator import Discriminator
from inference.figures import compute_fid, save_50
from inference.inference import inference

from trainer import Trainer

from arguments import get_args, get_model, get_dataset_fake, get_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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
        sampler=DistributedSampler(dataset, shuffle=False),
        # num_workers=0,
        drop_last=False
    )

def thread_main(rank, world_size, args, dataset_fake):
    # ======================================================================
    # device
    # ======================================================================
    ddp_setup(rank, world_size)

    
    # args.num_data_fake = args.num_data_fake//world_size
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'mps')
    print(device)
    
    # ======================================================================
    # model 
    # ======================================================================
    
    
    model = get_model(args).to(device)


    model = DDP(model, device_ids=[rank])
    trainer = Trainer(model, None, dataset_fake, args, None, None, None, device, dataloader_func=prepare_dataloader)
    
    
    try:
        trainer.warmup_fake_dataset(args, inference=True)
        destroy_process_group()
    except BaseException as e:
        destroy_process_group()
        raise e

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    
    assert batch_size > 0
    assert N > batch_size
    print(imgs[0][0][0])
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    # dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    #Get predictions
    preds = np.zeros((N, 1000))
    print(0, imgs.size()[0], batch_size)
    for i in range(0, imgs.size()[0], batch_size):
        if i+batch_size > imgs.size()[0]:
            # print(i+batch_size, imgs.size()[0])
            batch = imgs[i:].type(dtype)
        else:
            # print(i,i + batch_size)
            batch = imgs[i:i + batch_size].type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i:i + batch_size_i] = get_pred(batchv)
        # print(preds[i,0:20])
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

        print ("INCEPTION:",k+1,"out of", splits ,"steps complete", end='\r')
    print('')
    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__': 

    jsonpath = '/nas/users/tomislav/experiment1/generative_update/option/UNET_IMPROVED2_ATTPOOL_SMALL2/2024_02_26/00_44_36.json'
    plt_path = '/nas/users/tomislav/experiment1/generative_update/model/UNET_IMPROVED2_ATTPOOL_SMALL2/2024_02_26/00_44_36_89999.pth' # 42
    # /nas/users/tomislav/experiment1/generative_update/figure/UNET6/2023_09_13/22_09_08.png
    # Load args
    f = open(jsonpath)
    dict_obj = json.load(f)
    f.close

    pl.seed_everything(7777)
    # MAKE args
    parser = argparse.ArgumentParser()
    for key, value in dict_obj.items():
        parser.add_argument(f"--{key}", default=value)
    args = parser.parse_args()
    args.use_ema = False
    args.fake_data_warmup_at = -1
    world_size = torch.cuda.device_count()
    # args.size_batch = args.size_batch//world_size
    args.world_size = world_size
    args.fake_train_jumps = 1
    args.random_jumps = False
    # args.lr_fake += 3
    fid_min_samples = 50000
    args.batch_type = "inference"
    args.init_from = "gaussian"
    args.num_data_fake = (fid_min_samples //(args.size_batch*world_size))*args.size_batch*world_size
    dataset_fake = get_dataset_fake(args)

    args.load_model = plt_path
    
    
    print("Running parallel scripts.")
    print("fake_train_jumps=",args.fake_train_jumps)
    mp.spawn(thread_main, args=(world_size, args, dataset_fake), nprocs=world_size)
    
    imgs =  torch.tensor(dataset_fake.data).cpu().detach()

    # imgs =  torch.randn(imgs.size())
    print ("Calculating Inception Score...")
    print (inception_score(torch.clamp(imgs, min=-1, max=1), cuda=True, batch_size=32, resize=True, splits=10))

    