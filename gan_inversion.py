import os
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
from PIL import Image


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

# 생각해보니까, data_path도 달라져야함. 
# 이건 그냥 구조를 고정해두자.
def inversion(test_opts,checkpoint_type='',save=False):

    out_path_results = os.path.join(test_opts.exp_dir, checkpoint_type+'inference_results')
    os.makedirs(out_path_results, exist_ok=True)

    # update test options with options used during training
    if checkpoint_type == 'src':
        ckpt = torch.load(test_opts.source_checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts.update(vars(test_opts))
        opts.update({'checkpoint_path':test_opts.source_checkpoint_path})
        opts.update({'data_path':test_opts.source_data_path})
    elif checkpoint_type == 'tar':
        ckpt = torch.load(test_opts.target_checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts.update(vars(test_opts))
        opts.update({'checkpoint_path':test_opts.target_checkpoint_path})
        opts.update({'data_path':test_opts.target_data_path})
    else:
        ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts.update(vars(test_opts))
        opts.update({'checkpoint_path':test_opts.checkpoint_path})
    
    opts = Namespace(**opts)

    if opts.encoder_type in ENCODER_TYPES['pSp']:
        net = pSp(opts) 
    else:
        net = e4e(opts)

    net.eval()
    net.cuda()



    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()

    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    # get the image corresponding to the latent average
    avg_image = get_average_image(net, opts)

    if opts.dataset_type == "cars_encode":
        resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
    else:
        resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    global_i = 0
    global_time = []
    all_latents = {}
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch, result_latents = run_on_batch(input_cuda, net, opts, avg_image)
            # print(np.array(result_latents[1]).shape)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(input_batch.shape[0]):
            results = [tensor2im(result_batch[i][iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
            im_path = dataset.paths[global_i]

            # save step-by-step results side-by-side
            for idx, result in enumerate(results):
                save_dir = os.path.join(out_path_results, str(idx))
                os.makedirs(save_dir, exist_ok=True)
                result.resize(resize_amount).save(os.path.join(save_dir, os.path.basename(im_path)))

            # store all latents with dict pairs (image_name, latents)
            all_latents[os.path.basename(im_path)] = result_latents[i]

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))

    with open(stats_path, 'w') as f:
        f.write(result_str)

    # save all latents as npy file
    if save:
        np.save(os.path.join(test_opts.exp_dir, checkpoint_type+'latents.npy'), all_latents)
        return all_latents
    else:
        return all_latents

if __name__ == '__main__':
    test_opts = crossTestOptions().parse()
    inversion(test_opts)
