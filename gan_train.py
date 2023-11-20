import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split,Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import time
import math
from math import sqrt
from PIL import Image
import os
import random
from tqdm import tqdm

from util.normalization import image_normalization_extract
from util.util import data_loader_size
from util.metrics import psnr
from util.SSIM import ssim
from Dataset.image_dataset import imageDataset
from model.VDSR import VDSR
from model.SRCNN import SRCNN
from model.SRGAN import FeatureExtractor,ResidualBLock,GeneratorResNet,Discriminator
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# config
check_point = True
model = 'SRGAN'  # 대문자
project_name = 'test'
save_model_file_name='SRGAN_TEST11_16_thermal_image'

#path
# image_path = r"C:\Users\microsoft\Desktop\이삿짐\code\songsan_project\test\\"
# image_path=r"C:\Users\microsoft\Desktop\이삿짐\code\songsan_project\New_Sample\원천데이터"
# image_path = '/content/drive/MyDrive/image_dataset/test_sensor'
# image_path = '/content/test_data/set_data'
image_path_separate= '/content/dataset_8'        # 개별지정
HR_path = f'{image_path_separate}/hr'            # 개별지정
LR_path = f'{image_path_separate}/lr'            # 개별지정
bicubic_path = f'{image_path_separate}/bicubic'  # 개별지정
one_path = False # 폴더 하나로 나누기
image_path = '/content/drive/MyDrive/image_dataset/train_dataset290'
save_path = '/content/drive/MyDrive/SR code/Super_Resolution_model'

#parametor
image_size = (32,32)
Downsizing_factor = 4
num_epochs =16
batch_size = 32
b1 = 0.5
b2 =0.999
lr = 0.0001


# normalization
# nor_test_trans = transforms.Compose([transforms.Resize(image_size),
#                             transforms.ToTensor()])
# nor_test_data_set = torchvision.datasets.ImageFolder(root =image_path,
# 					transform = nor_test_trans,)
# nor_mean, nor_std = image_normalization_extract(nor_test_data_set)
nor_mean ,nor_std = ((0.5,0.5,0.5),(0.5,0.5,0.5))

# image Data load
#고해상도
if one_path == True:
  HR_trans = transforms.Compose([transforms.Resize(image_size,interpolation=transforms.InterpolationMode.BICUBIC),
                              transforms.ToTensor(),
                              transforms.Normalize(nor_mean,nor_std)
                              ])
  HR_data_set = torchvision.datasets.ImageFolder(root =image_path,
            transform = HR_trans,
                      target_transform=None)
  #저해상도
  LR_trans = transforms.Compose([transforms.Resize((image_size[0]//Downsizing_factor,image_size[1]//Downsizing_factor)),
                              transforms.ToTensor(),
                              transforms.Normalize(nor_mean, nor_std),
                              ])
  LR_data_set = torchvision.datasets.ImageFolder(root =image_path,
            transform = LR_trans,
                      target_transform=None)
  #저해상도-bicubic 보간
  LR_BICUBIC_trans = transforms.Compose([transforms.Resize((image_size[0]//Downsizing_factor,image_size[1]//Downsizing_factor)),
                              transforms.Resize(image_size,interpolation=transforms.InterpolationMode.BICUBIC),
                              transforms.ToTensor(),
                              transforms.Normalize(nor_mean, nor_std),
                              ])
  LR_BICUBIC_data_set = torchvision.datasets.ImageFolder(root =image_path,
            transform = LR_BICUBIC_trans, target_transform=None)
else:
  #개별 폴더 로드
  trans = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(nor_mean, nor_std)])

  HR_data_set = torchvision.datasets.ImageFolder(root =HR_path,
            transform = trans, target_transform=None)
  LR_data_set = torchvision.datasets.ImageFolder(root =LR_path,
            transform = trans, target_transform=None)
  LR_BICUBIC_data_set = torchvision.datasets.ImageFolder(root =bicubic_path,
            transform = trans, target_transform=None)


# train valid test
validation_split = 0.2
test_split = 0.05

dataset_size=len(HR_data_set)
indices = list(range(dataset_size))
valid_split = int(np.floor(validation_split * dataset_size))
test_split = int(np.floor(test_split * dataset_size))

np.random.seed(98)
np.random.shuffle(indices)

train_idx,valid_idx,test_idx = indices[valid_split+test_split:],indices[:valid_split],indices[valid_split:valid_split+test_split]

#Dataset -> Dataloader
total_ds=imageDataset(LR_data_set, HR_data_set,LR_BICUBIC_data_set)

new_train_loader = DataLoader(Subset(total_ds,train_idx), batch_size=batch_size,shuffle=None)
new_valid_loader = DataLoader(Subset(total_ds,valid_idx), batch_size=batch_size,shuffle=None)
new_test_loader = DataLoader(Subset(total_ds,test_idx), batch_size=1,shuffle=None)


#학습단계
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
cuda = torch.cuda.is_available()

generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(3, *image_size))
feature_extractor = FeatureExtractor()
feature_extractor.eval()

Gan_loss = torch.nn.BCELoss()
content_loss = torch.nn.L1Loss()

if cuda:
  generator = generator.cuda()
  discriminator = discriminator.cuda()
  feature_extractor=feature_extractor.cuda()
  Gan_loss = Gan_loss.cuda()
  content_loss = content_loss.cuda()
#
optimizer_G = torch.optim.Adam(generator.parameters(),lr=lr,betas=(b1,b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(b1,b2))
start = time.time()

generator_losses = []
discriminator_losses = []
metrics_valid = pd.DataFrame(data={'epoch':[],'PSNR':[],'SSIM':[]})
epoch_start = 0

if check_point ==True:
  check_point = torch.load(f'{save_path}/check_point/check_point_model_{model}_{project_name}.pt')
  generator.load_state_dict(check_point['generator'])
  optimizer_G.load_state_dict(check_point['optimizer_G'])
  discriminator.load_state_dict(check_point['discriminator'])
  optimizer_D.load_state_dict(check_point['optimizer_D'])
  generator_losses = check_point['generator_loss']
  discriminator_losses = check_point['discriminator_loss']
  epoch_start = check_point['epoch']
  metrics_valid = pd.read_csv(f'{save_path}/save_metrics/{model}_{project_name}_result.csv',index_col = 0)


for Epoch in range(epoch_start,num_epochs ):
    print(f'Epoch {Epoch+1} of {num_epochs}')
    discriminator.train()
    generator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0
    for img in tqdm(new_train_loader):

        image = Variable(img[0].type(Tensor)).to(device)
        label = Variable(img[1].type(Tensor)).to(device)
        #image = img[0].type(Tensor).to(device)
        #label = img[1].type(Tensor).to(device)

        #valid = torch.ones((image.size()[0],*discriminator.output_shape), requires_grad =False).float().cuda()
        #fake = torch.zeros((image.size()[0],*discriminator.output_shape), requires_grad =False).float().cuda()
        valid = Variable(torch.Tensor(np.ones((image.size(0)))), requires_grad=False).cuda()
        fake = Variable(torch.Tensor(np.zeros((image.size(0)))), requires_grad=False).cuda()


        #train generators
        optimizer_G.zero_grad()

        gen_hr = generator(image)

        loss_GAN = Gan_loss(discriminator(gen_hr),valid)

        #Content_loss
        gen_feature = feature_extractor(gen_hr)
        real_feature = feature_extractor(label)
        loss_Content = content_loss(gen_feature,real_feature.detach())

        #total loss
        loss_G = loss_Content + 0.001* loss_GAN

        loss_G.backward()
        optimizer_G.step()

        #train Discriminator
        optimizer_D.zero_grad()

        loss_real = Gan_loss(discriminator(label),valid)
        loss_fake = Gan_loss(discriminator(gen_hr.detach()),fake)

        #total loss
        loss_D = (loss_real + loss_fake) /2

        loss_D.backward()
        optimizer_D.step()

        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()
    final_running_loss_G = running_loss_G/len(new_train_loader.sampler)
    generator_losses.append((Epoch,final_running_loss_G))
    final_running_loss_D = running_loss_D/len(new_train_loader.sampler)
    discriminator_losses.append((Epoch,final_running_loss_D))
    end = time.time()
    print(f'Train loss_G : {final_running_loss_G:.4f}, Train loss_D : {final_running_loss_D:.10f} ,Time : { end-start:.2f} sec')
    # check point
    check_point = {'generator' : generator.state_dict(), 'optimizer_G' : optimizer_G.state_dict(), 'discriminator': discriminator.state_dict(),
                   'optimizer_D': optimizer_D.state_dict(), 'generator_loss':generator_losses, 'discriminator_loss':discriminator_losses,
                   'epoch':Epoch }
    torch.save(check_point, f'{save_path}/check_point/check_point_model_{model}_{project_name}.pt')


    # validation
    if Epoch % 2 == 0:
      with torch.no_grad():
        generator.eval()
        discriminator.eval()
        img_count=0
        psnr_avg = 0.0
        ssim_avg=0.0
        # ssim = 0.0
        for img in new_valid_loader:
          LR_image = img[0].to(device)
          label = img[1].to(device)
          sr_image = generator(LR_image)
          psnr_avg +=psnr(label,sr_image)
          ssim_avg +=ssim(label,sr_image).item()
          img_count+=1
          LR_image = nn.functional.interpolate(LR_image,scale_factor=4)
          sr_image = make_grid(sr_image,nrow=1, normalize=True)
          LR_image = make_grid(LR_image, nrow=1, normalize=True)
          img_grid = torch.cat((LR_image,sr_image),-1)
          save_image(img_grid,f'{save_path}/save_img/{model}/{model}_{project_name}_{Epoch}.png',normalize=False)
        psnr_avg /= img_count
        ssim_avg /= img_count
        print(f'PSNR_valid:{psnr_avg:.4f}, SSIM_valid:{ssim_avg:.4f}' )
        metrics_valid_update = pd.DataFrame(data={'epoch':[Epoch],'PSNR':[psnr_avg],'SSIM':[ssim_avg]})
        metrics_valid = pd.concat([metrics_valid,metrics_valid_update])
        metrics_valid.to_csv(f'{save_path}/save_metrics/{model}_{project_name}_result.csv')

#model save
torch.save(generator.state_dict(), f'{save_path}/save_model/SRGAN/{save_model_file_name}.pt')













