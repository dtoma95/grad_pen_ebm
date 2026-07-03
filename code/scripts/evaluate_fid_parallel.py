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


if __name__ == '__main__': 

    jsonpath = '/option/cifar10_resnet.json'
    plt_path = '/model/cifar10_resnet.pth' # 42
    
    # Load args
    print ("Load args from:", jsonpath)
    f = open(jsonpath)
    dict_obj = json.load(f)
    f.close

    # pl.seed_everything(7777)
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
    args.refresh_rate = 1
    args.random_jumps = False
    # args.lr_fake -= 0.5
    fid_min_samples = 50000
    args.batch_type = "inference"
    args.init_from = "gaussian"
    # args.name_dataset = "cifar10pt"
    args.num_data_fake = (fid_min_samples //(args.size_batch*world_size))*args.size_batch*world_size
    dataset_fake = get_dataset_fake(args)

    args.load_model = plt_path
    
    
    print("Running parallel scripts.")
    print("fake_train_jumps=",args.fake_train_jumps)
    mp.spawn(thread_main, args=(world_size, args, dataset_fake), nprocs=world_size)
    
    fake_all = []
    for i in range(0, args.num_data_fake, args.size_batch):
        fake_all.append(dataset_fake.data[i:i+args.size_batch])

    dataset = get_dataset(args)
    real_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.size_batch, shuffle=True, drop_last=True)
    fid = compute_fid(real_dataloader, fake_all, args.size_batch, sample_size=len(dataset))#args.num_data_fake)
    print(fid)

    images_50 = dataset_fake.data[0:50].detach().cpu().clone().numpy()
    images_50 = np.clip(np.array([images_50]), -1, 1)
    args.length_time = 1
    label_fake  = torch.randint(0, 9, (50, 1))

    save_50(images_50, '../inference_png/parallel_inference_results_50.png', label_fake, args)

    images_50 = dataset_fake.data[50:100].detach().cpu().clone().numpy()
    images_50 = np.clip(np.array([images_50]), -1, 1)
    save_50(images_50, '../inference_png/parallel_inference_results_502.png', label_fake, args)

    images_50 = dataset_fake.data[100:dataset_fake.data.size()[0]].detach().cpu().clone().numpy()
    images_50 = np.clip(np.array([images_50]), -1, 1)
    label_fake  = torch.randint(0, 9, (dataset_fake.data.size()[0]-100, 1))
    save_50(images_50, '../inference_png/parallel_inference_results_503.png', label_fake, args)

    logf = open("fid_cifar10.log", "w")
    logf.write(plt_path + "  " + str(fid))
    logf.close()
    print(args.weighting_type)
    





