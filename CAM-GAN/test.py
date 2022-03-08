import argparse
import os
from os import path
import copy
from tqdm import tqdm
import torch
from torch import nn
from gan_training import utils
import numpy as np
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models
)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='2,3' 
# Arguments
parser = argparse.ArgumentParser(
    description='Test a trained GAN and create visualizations.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)

# Shorthands
nlabels = config['data']['nlabels']
out_dir = config['training']['out_dir']
batch_size = config['test']['batch_size']
sample_size = config['test']['sample_size']
sample_nrow = config['test']['sample_nrow']
checkpoint_dir = path.join(out_dir, 'chkpts')
img_dir = path.join(out_dir, 'test', 'img')
img_all_dir = path.join(out_dir, 'test', 'img_all')

# Creat missing directories
if not path.exists(img_dir):
    os.makedirs(img_dir)
if not path.exists(img_all_dir):
    os.makedirs(img_all_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

# Get model file
model_file = config['test']['model_file']

# Models
device = torch.device("cuda:0" if is_cuda else "cpu")

generator, discriminator = build_models(config)
print(generator)
print(discriminator)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
)

# Test generator
if config['test']['use_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Distributions
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                  device=device)

# Evaluator
NNN=20000
train_dataset, nlabels = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size'],
    lsun_categories=config['data']['lsun_categories_train']
)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True
)
x_real_FID, _ = utils.get_nsamples(train_loader, NNN)
evaluator = Evaluator(generator_test, zdist, ydist,
                          batch_size=batch_size, inception_nsamples=NNN,device=device,
                          fid_real_samples=x_real_FID,  fid_sample_size=NNN)



#path to the checkpoints
model_path = '/raid/sakshi/output/flowers/chkpts/model.pt'


import glob
model_list = [f for f in glob.glob(model_path)]
print(model_list)
# Load checkpoint if existant


# Inception score
inception_mean_all = []
inception_std_all = []
fid_all = []


for m in model_list:
    load_dict = checkpoint_io.load(m)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)
    
    if config['test']['compute_inception']:
        print('Computing inception score...')
        inception_mean, inception_std, fid = evaluator.compute_inception_score()
        inception_mean_all.append(inception_mean)
        inception_std_all.append(inception_std)
        fid_all.append(fid)
        print('test it %d: IS: mean %.2f, std %.2f, FID: mean %.2f' % (
                    it, inception_mean, inception_std, fid))

        FID = np.stack(fid_all)
        Inception_mean = np.stack(inception_mean_all)
        Inception_std = np.stack(inception_std_all)





    '''sio.savemat(out_path + DATA + 'base_FID_IS.mat', {'FID': FID,
                                                       'Inception_mean': Inception_mean,
                                                       'Inception_std': Inception_std})'''
                                                       
                                                      
print(FID)

# Samples
'''print('Creating samples...')
ztest = zdist.sample((sample_size,))
x = evaluator.create_samples(ztest)
utils.save_images(x, path.join(img_all_dir, '%08d.png' % it),
                  nrow=sample_nrow)
if config['test']['conditional_samples']:
    for y_inst in tqdm(range(nlabels)):
        x = evaluator.create_samples(ztest, y_inst)
        utils.save_images(x, path.join(img_dir, '%04d.png' % y_inst),
                          nrow=sample_nrow)'''
