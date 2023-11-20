import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg19

from math import sqrt


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)

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

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height/2**4), int(in_width/2**4) # size 차이를 확인해봐야 할듯
        self.output_shape = (1,patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers=[]
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters,out_filters, kernel_size = 3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64,128,256,512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i==0)))
            in_filters = out_filters

        #Dense layer 생략, fully connected layer는 파라미터를 급격하게 증가시켜 모델이 커져 학습이 어려움
        #layers.append(nn.Conv2d(out_filters,1,kernel_size=3,stride=1,padding=1))
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Conv2d(512,1024, kernel_size=1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(1024,1,kernel_size=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
      batch_size = img.size(0)
      return torch.sigmoid(self.model(img).view(batch_size))