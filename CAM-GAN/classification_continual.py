import argparse
import os, sys
from os import path
import time
import copy
import torch
from torch import nn
import numpy as np
import random
import shutil
import torchvision.models as models
from gan_training.distributions import get_ydist
import torch.nn.functional as F
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='4'
import pdb

def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(999)
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints_lifelong import CheckpointIO
from gan_training.inputs import get_dataset
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler,
)
from EWC import Net
import scipy.io as sio
ce_loss = nn.CrossEntropyLoss()
#main_path = './code_GAN_Memory/'


data_transforms = {
    'train1': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
out_dir='/raid/srijith/sakshi/'
logger = Logger(
    log_dir=path.join(out_dir, 'logs'),
    img_dir=path.join(out_dir, 'imgs'),
)

def select_task_path(task_id, is_test=False):
    if not is_test:
        if task_id == 0:
            train_path = '/raid/srijith/sakshi/imagenet/fish/'
        elif task_id == 1:
            train_path = '/raid/srijith/sakshi/imagenet/bird/'
        elif task_id == 2:
            train_path = '/raid/srijith/sakshi/imagenet/snake_new/'
        elif task_id == 3:
            train_path = '/raid/srijith/sakshi/imagenet/dog/'
        elif task_id == 4:
            train_path = '/raid/srijith/sakshi/imagenet/butterfly/'
        elif task_id == 5:
            train_path = '/raid/srijith/sakshi/imagenet/insect/'
        return train_path
    elif is_test:
        if task_id == 0:
            train_path = '/raid/srijith/sakshi/imagenet' + '/test/fish/'
        elif task_id == 1:
            train_path = '/raid/srijith/sakshi/imagenet'+ '/test/bird/'
        elif task_id == 2:
            train_path = '/raid/srijith/sakshi/imagenet' + '/test/snake_new/'
        elif task_id == 3:
            train_path = '/raid/srijith/sakshi/imagenet' + '/test/dog/'
        elif task_id == 4:
            train_path = '/raid/srijith/sakshi/imagenet' + '/test/butterfly/'
        elif task_id == 5:
            train_path = '/raid/srijith/sakshi/imagenet' + '/test/insect/'
        return train_path


        
        
# -------------------------------------------------------------
# -------------------------------------------------------------

batch_size= 36
N_task = 6
N_epoch = 1
N_labels = N_task * 6
do_method = 'Style_transfer' #'EWC' #'MeRGAN'   # GAN_Memory   MeRGAN   'Joint'  'Joint1'
# -------------------------------------------------------------
# -------------------------------------------------------------



def evalu_my(model, test_loader, test_task=-1):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if test_task>=0:
            target = target+ test_task*6
        #print(target)
        output = model(data)
        test_loss += ce_loss(output, target).data # sum up batch loss
        _, pred = output.data.max(1, keepdim=True) # get the index of the max log-probability
        #print(target,pred) self.module_dict.update(kwargs)
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct.float() / len(test_loader.dataset)
    print('\nTest set task {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
    test_task, test_loss, correct, len(test_loader.dataset),test_acc))
    return test_acc


#train_path_all = 'F:/download_data/Image_DATA/ImageNet_GANmemory/'
#model_path_all = 'F:/RUN_CODE_OUT/OWM/'


test_path = '/raid/srijith/sakshi/imagenet/test_combined/'
is_cuda=True
device = torch.device("cuda:0" if is_cuda else "cpu")
config_path = '/raid/srijith/sakshi/GAN_stability/configs/fish_old.yaml'
config = load_config(config_path, 'configs/default.yaml')


test_dataset = datasets.ImageFolder(os.path.join(test_path), data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
class_names = test_dataset.classes
print(class_names)


classify_my = Net(nlabels=N_labels, device=device).to(device)
c_optimizer = torch.optim.Adam(classify_my.params, lr=5*1e-5)

checkpoint_dir='/raid/srijith/sakshi/GAN_stability/'

checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)


if do_method == 'Style_transfer':
    config['generator']['name'] = 'resnet4_adapter'
    config['discriminator']['name'] = 'resnet4_adapter'
    config['data']['nlabels'] = 6

    generator, _ = build_models(config)
    generator = generator.to(device)
    generator = nn.DataParallel(generator)

   

    checkpoint_io.register_modules(
    generator=generator,
    )

def select_model(task_id ):
    if do_method == 'Style_transfer':
        if task_id == 0:
            model_path = '/raid/srijith/sakshi/GAN_stability/output/fish/chkpts/model_00034999.pt'
            
        elif task_id == 1:
            model_path = '/raid/srijith/sakshi/GAN_stability/output/bird/chkpts/model_00022499.pt'
           
        elif task_id == 2:
            model_path = '/raid/srijith/sakshi/GAN_stability/output/snake_new/chkpts/model_00037499.pt'

        elif task_id == 3:
            model_path = '/raid/srijith/sakshi/GAN_stability/output/dog/chkpts/model_00052499.pt'
           
        elif task_id == 4:
            model_path='/raid/srijith/sakshi/GAN_stability/output/butterfly/chkpts/model_00039999.pt'
    return model_path
            

    
    

    
        


        
    #print(generator)
    '''load_dir = './pretrained_model/'
    DATA_FIX = 'CELEBA'
    dict_G = torch.load(load_dir + DATA_FIX + 'Pre_generator')
    generator = model_equal_part_embed(generator, dict_G)
    generator = load_model_norm(generator)

    task_name = ['fish', 'bird', 'snake', 'dog', 'butterfly', 'monkey']
    for task_id in range(6):
        model_file = main_path + '/results/imagenet_' + task_name[task_id] + '/models/'
        dict_G = torch.load(model_file + task_name[task_id] + '_%08d_Pre_generator' % 59999)
        generator = model_equal_part_embed(generator, dict_G)
        generator(task_id=task_id, UPDATE_GLOBAL=True)'''


acc_all_i = [[],[],[],[],[],[]]
acc_all = []
save_path = '/raid/srijith/sakshi/classification_result/'

for n, param in classify_my.feat.named_parameters():
    param.requires_grad = True

file_name= '/raid/srijith/sakshi/classification_result/Accuracy.txt'

for task_id in range(6):
    # prepare dataloader
    if(task_id>0):
        lr=1e-5
    if do_method == 'Joint':
        n_c = batch_size*(task_id+1)
        train_path = select_task_path_joint(task_id)
        train_dataset = datasets.ImageFolder(os.path.join(train_path), data_transforms['train'])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_c,
                                                   shuffle=True, num_workers=0)
    else:
        n_c = batch_size
        n_p = batch_size

        train_path = select_task_path(task_id)
        train_dataset = datasets.ImageFolder(os.path.join(train_path), data_transforms['train'])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_c,
                                                   shuffle=True, num_workers=0)


    

    if do_method == 'Style_transfer':
            

        classifier_old = copy.deepcopy(classify_my).eval()
        IT = 0
        acc_task = []
        acc_task_i = []
        for epoch in range(N_epoch):
            for (x_cur, y_cur) in train_loader:
                y_cur = task_id*6 + y_cur
                classify_my.train()
                x_cur, y_cur = x_cur.to(device), y_cur.to(device)
                c_optimizer.zero_grad()

                x_replay = []
                y_replay = []
                x_replay.append(x_cur)
                y_replay.append(y_cur)
                if task_id > 0:
                    with torch.no_grad():
                        if do_method=='Style_transfer':
                            nlabels = 6


                            y_sample = get_ydist(nlabels, device=device)
                            z_sample = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                                                 device=device)
                            y_replay0 = y_sample.sample((n_p,)).to(device)
                            z = z_sample.sample((n_p,)).to(device)
                            for i_t in range(task_id):

                                model_path=select_model(i_t)
                                load_dict = checkpoint_io.load(model_path)
                                it = load_dict.get('it', -1)
                                epoch_idx = load_dict.get('epoch_idx', -1)
                                

                                
                                x_replay0= generator(z, y_replay0)
                                logger.add_imgs(x_replay0, 'generative_replay', it)
                                x_replay0 = F.interpolate(x_replay0, 224, mode='bilinear')
                                mu_0 = torch.tensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(
                                    2).unsqueeze(3)
                                st_0 = torch.tensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(
                                    2).unsqueeze(3)
                                x_replay0 = ((x_replay0 + 1.0) / 2.0 - mu_0) / st_0
                                x_replay.append(x_replay0)
                                y_replay.append(6 * i_t + y_replay0)
                                # y_replay.append(y_replay0)
                                
                                
                x_replay = torch.cat(x_replay)
                y_replay = torch.cat(y_replay)

                logits_S = classify_my(x_replay.detach())
                loss_replay = ce_loss(logits_S, y_replay)
                print("training_loss"+str(loss_replay.item()))
                loss_replay.backward()
                c_optimizer.step()
                IT +=1
                if IT%50==0:
                    with torch.no_grad():
                        test_acc_i=0.0
                        for i_t in range(task_id+1):
                            print(i_t)
                            test_path_i = select_task_path(i_t, is_test=True)
                            test_dataset_i = datasets.ImageFolder(os.path.join(test_path_i), data_transforms['test'])
                            test_loader_i = torch.utils.data.DataLoader(test_dataset_i, batch_size=batch_size,
                                                                        shuffle=False, num_workers=0)
                            class_names = test_dataset_i.classes
                            print(class_names)

                            test_acc_i = evalu_my(classify_my, test_loader_i, test_task=i_t)
                            acc_all_i[i_t].append(test_acc_i.data.cpu())
                        test_acc = evalu_my(classify_my, test_loader)
                        with open(file_name, 'a') as f:
                            f.write("over_all Accuracy_after_task_%5f %5f\r\n" % (task_id,test_acc))
                          
                        
                        acc_all.append(test_acc.data.cpu())
                        print('\nTest set task {}/ epoch {}: Accuracy: ({:.5f}% / {:.5f}%) \n'.format(
                            task_id, epoch, test_acc_i, test_acc))

    


ACC_all_0 = np.stack(acc_all_i[0])
ACC_all_1 = np.stack(acc_all_i[1])
ACC_all_2 = np.stack(acc_all_i[2])
ACC_all_3 = np.stack(acc_all_i[3])
ACC_all_4 = np.stack(acc_all_i[4])
ACC_all_5 = np.stack(acc_all_i[5])
ACC_all = np.stack(acc_all)
print(acc_all)
sio.savemat(save_path + do_method + '_insect9_res18_test_acc_layerAll_lambda_OPT_ALL2.mat', {'ACC_all_0': ACC_all_0,
                                                               'ACC_all_1': ACC_all_1,
                                                               'ACC_all_2': ACC_all_2,
                                                               'ACC_all_3': ACC_all_3,
                                                               'ACC_all_4': ACC_all_4,
                                                               'ACC_all_5': ACC_all_5,
                                                               'ACC_all': ACC_all,})









