import numpy as np
import os
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from PIL import Image
import random


sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from options.cross_test_options import crossTestOptions
from models.psp import pSp
from models.e4e import e4e
from utils.model_utils import ENCODER_TYPES
from utils.common import tensor2im
from utils.inference_utils import run_on_batch, get_average_image
from gan_inversion import inversion

def load_npy(npy_path):
    return np.load(npy_path, allow_pickle=True)

def load_generator(test_opts, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts.update({'checkpoint_path':ckpt_path})
    opts = Namespace(**opts)

    if opts.encoder_type in ENCODER_TYPES['pSp']:
        net = pSp(opts).decoder 

    net.eval()
    net.cuda()
    return net

if __name__ =='__main__':
    test_opts = crossTestOptions().parse()
    if test_opts.load_numpy:
        source_latent = load_npy('/home/work/NaverWebtoonSide/restyle-encoder/experiment/source_latent.npy')
        target_latent = load_npy('/home/work/NaverWebtoonSide/restyle-encoder/experiment/latents.npy')
    else:
        source_latent = inversion(test_opts,'src')
        target_latent = inversion(test_opts,'tar')

    # preprocess latent codes
    source_latent = torch.tensor(np.array(list(source_latent.item().values())[0])[-1]).cuda()

    # Random k( default : 50) sampling from specific character ID cartoon dataset
    target_latent_imgs = np.random.choice(list(target_latent.item().keys()),test_opts.k_sampling)

    # Averaging Latent Codes which are specific cartoon character ID ( Papaer Section4.1 Formula (1) )
    target_latent_list = []
    for i in target_latent_imgs:
        t = target_latent.item()[i]
        target_latent_list.append(t)

    target_latent = torch.tensor(np.mean(target_latent_list,axis=0)[-1]).cuda() # Because of Restyle Encoder which has 5 iteration training, I select 5th Iteration latent code
    
    # update test options with options used during training
    src_generator = load_generator(test_opts, test_opts.source_checkpoint_path)
    tar_generator = load_generator(test_opts, test_opts.target_checkpoint_path)


    # layer swapping
    resolution_level = {8:0, 16:2, 32:4, 64:6,128:8,256:10,512:12, 1024:14}
    
    for key in list(filter(lambda x : x>=test_opts.layer_swap_resolution, resolution_level.keys())):
        index = resolution_level[key]
        src_generator.convs[index] = tar_generator.convs[index]
        src_generator.convs[index+1] = tar_generator.convs[index+1]
        src_generator.to_rgbs[(index-2)//2] = tar_generator.to_rgbs[(index-2)//2]
    layer_swap_generator = src_generator
    
    # TRGB Replacement, Style Mixing, Generation
    result = layer_swap_generator.cross_forward(source_latent,target_latent) 
    result = tensor2im(result[0])
    result.save(test_opts.out_path)

