import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split,Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision
from torchvision import transforms

import os


#path
image_path = '/content/test_data/set_data/test/' # 원본
save_path = '/content/dataset_6' # 변형 이미지 저장경로
image_size = (64,64)
Downsizing_factor = 8


nor_mean ,nor_std = ((0.5,0.5,0.5),(0.5,0.5,0.5))

# image Data load
#고해상도
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
					transform = LR_BICUBIC_trans,
                    target_transform=None)


os.mkdir(f'{save_path}')
os.makedirs(f'{save_path}/lr/image')
os.makedirs(f'{save_path}/hr/image')
os.makedirs(f'{save_path}/bicubic/image')
LR_dataloader = DataLoader(LR_data_set, batch_size=1, shuffle=None)
HR_dataloader = DataLoader(HR_data_set, batch_size=1, shuffle=None)
BICUBIC_dataloader = DataLoader(LR_BICUBIC_data_set, batch_size=1, shuffle=None)
inverse_mean = [-m/s for m, s in zip(nor_mean, nor_std)]
inverse_std = [1/s for s in nor_std]
nor_trans=transforms.Normalize(inverse_mean, inverse_std)
for num, (image,labels) in enumerate(LR_dataloader):
  transforms.ToPILImage()(nor_trans(image[0])).save(f'{save_path}/lr/image/{num}.png')
  print(num)
for num, (image,labels) in enumerate(HR_dataloader):
  transforms.ToPILImage()(nor_trans(image[0])).save(f'{save_path}/hr/image/{num}.png')
  print(num)
for num, (image,labels) in enumerate(BICUBIC_dataloader):
  transforms.ToPILImage()(nor_trans(image[0])).save(f'{save_path}/bicubic/image/{num}.png')
  print(num)

