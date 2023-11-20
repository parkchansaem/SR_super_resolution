import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,64,9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64,32,1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32,3,5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))  
