import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import torchvision 
from torchvision import transforms
from torchvision.utils import save_image

import time
import math
from math import sqrt
from PIL import Image
import os
import random

from util.normalization import image_normalization_extract
from Dataset.image_dataset import imageDataset
from model.VDSR import VDSR
from model.SRCNN import SRCNN
from train import train,validate
from visualization.loss import Loss_and_metrics_visualization
from visualization.output import output_visualization

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(98) # Seed 고정

device = torch.device('cpu')
#path
image_path = r"C:\Users\microsoft\Desktop\이삿짐\code\songsan_project\test\\"
save_path=''

#parameter
image_size = (320,160)
num_epochs =100
batch_size = 4

# normalization
nor_test_trans = transforms.Compose([transforms.Resize(image_size), 
                            transforms.ToTensor()])
nor_test_data_set = torchvision.datasets.ImageFolder(root =image_path, 
					transform = nor_test_trans,)
nor_mean, nor_std = image_normalization_extract(nor_test_data_set)

# image Data load
HR_trans = transforms.Compose([transforms.Resize(image_size,interpolation=transforms.InterpolationMode.BICUBIC), 
                            transforms.ToTensor(),
                            transforms.Normalize(nor_mean,nor_std)
                            ])
HR_data_set = torchvision.datasets.ImageFolder(root =image_path, 
					transform = HR_trans,
                    target_transform=None)

LR_trans = transforms.Compose([transforms.Resize((image_size[0]//4,image_size[1]//4)),
                            transforms.Resize(image_size,interpolation=transforms.InterpolationMode.BICUBIC), 
                            transforms.ToTensor(),
                            transforms.Normalize(nor_mean, nor_std),
                            ])
LR_data_set = torchvision.datasets.ImageFolder(root =image_path, 
					transform = LR_trans,
                    target_transform=None)

# train valid test
validation_split = 0.2

dataset_size=len(HR_data_set)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.seed(98)
np.random.shuffle(indices)

train_idx,valid_idx = indices[split:],indices[:split]

#Dataset -> Dataloader
total_ds=imageDataset(LR_data_set, HR_data_set)

new_train_loader = DataLoader(Subset(total_ds,train_idx), batch_size=batch_size,shuffle=None)
new_valid_loader = DataLoader(Subset(total_ds,valid_idx), batch_size=batch_size,shuffle=None)

#model = SRCNN().to(device)
model = VDSR().to(device)
loss_func = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                        step_size = num_epochs//4,
                                        gamma = 0.1,
                                        verbose = False )

train_loss, val_loss = [],[]
train_psnr, val_psnr = [],[]
train_ssim, val_ssim = [],[]
best_loss= 0.0
best_model = None
start = time.time()
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1} of {num_epochs}')
    train_epoch_loss, train_epoch_psnr,train_epoch_ssim = train(model=model, 
                                                                data_dl=new_train_loader, 
                                                                optimizer=optimizer,
                                                                device=device,
                                                                loss_func=loss_func)
    scheduler.step()
    print("lr: ", optimizer.param_groups[0]['lr'])
    val_epoch_loss, val_epoch_psnr,val_epoch_ssim = validate(model=model, 
                                                             data_dl=new_valid_loader,
                                                             device=device,
                                                             loss_func=loss_func,
                                                             epoch=epoch
                                                             )

    train_loss.append(train_epoch_loss)
    train_psnr.append(train_epoch_psnr)
    train_ssim.append(train_epoch_ssim)
    val_loss.append(val_epoch_loss)
    val_psnr.append(val_epoch_psnr)
    val_ssim.append(val_epoch_ssim)
    
    end = time.time()
    print(f'Train loss : {train_epoch_loss:.4f}, Valid loss : { val_epoch_loss:.3f}')
    print(f'Train SSIM : {train_epoch_ssim:.4f}, Valid SSIM : { val_epoch_ssim:.3f}')
    print(f'Train PSNR : {train_epoch_psnr:.4f}, Valid PSNR : { val_epoch_psnr:.3f}, Time : { end-start:.2f} sec')

    if best_loss < val_epoch_psnr:
            best_loss = val_epoch_psnr
            best_model = model

# Loss and metrics visualization
Loss_and_metrics_visualization(train_loss,train_psnr,train_ssim,val_loss,val_psnr,val_ssim)

#inference
for img , label in new_valid_loader:
    img= img[0]
    label = label[0]
    break

model.eval()
with torch. no_grad():
    img_ = img.unsqueeze(0)
    img_ = img_.to(device)
    output = model(img_)
    output = output.squeeze(0)
output_visualization(img,output,label,nor_mean,nor_std)
#torch.save(model.state_dict(), './vdsr_save_model/vdsr_epoch50_model.pt')
torch.save(model.state_dict(), f'{save_path}/vdsr_epoch50_model.pt')