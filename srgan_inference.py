import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from math import sqrt


# 정규화, 텐서 변환 
nor_mean ,nor_std = ((0.5,0.5,0.5),(0.5,0.5,0.5))
inverse_mean = [-m/s for m, s in zip(nor_mean, nor_std)]
inverse_std = [1/s for s in nor_std]
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(nor_mean,nor_std)])
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=inverse_mean, std=inverse_std),
    transforms.ToPILImage()  # 텐서를 PIL 이미지로 변환
])

# SRGAN 모델
class ResidualBLock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBLock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_features,0.8),
            nn.PReLU(),
            nn.Conv2d(in_features,in_features,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self,x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels = 3 , out_channels = 3, n_residual_blocks = 16):
        super(GeneratorResNet, self).__init__()

        # first layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=9,stride=1, padding=4),nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBLock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # second conv layer
        self.conv2 = nn.Sequential(nn.Conv2d(64,64, kernel_size=3, stride = 1 , padding=1), nn.BatchNorm2d(64,0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling +=[
                nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),   # 256/(upscale_fetor=2**2)-> output_feature =64
                nn.PReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())
        # 왜 tanH활성화 함수 사용?
        # 범위를 매치시키는 것이 목표이다. 실제 이미지는 -1,1 사이 값을 가지므로 tanh를 활용하여 범위를 동일하게 만들었다.

    def forward(self, x):
        out1= self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1,out2) #elementwise sum, skip connection
        out = self.upsampling (out)
        out = self.conv3(out)
        return out

# path
save_path= '/content'
save_file_name ='1'
model_path =r'/content/drive/MyDrive/SR code/Super_Resolution_model/save_model/SRGAN/SRGAN_TEST11_16_thermal_image.pt'
image_path = r'/content/dataset_6/hr/image/4.png'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = GeneratorResNet()
generator.load_state_dict(torch.load(model_path,map_location=device))
generator=generator.to(device)

img = Image.open(r'/content/dataset_6/hr/image/4.png')
img = trans(img)
generator.eval()
with torch.no_grad():
  output = img.unsqueeze(0).to(device)
  output = generator(output).squeeze(0)
generate_img=inverse_transform(output)
generate_img.save(f'{save_path}/{save_file_name}.png')