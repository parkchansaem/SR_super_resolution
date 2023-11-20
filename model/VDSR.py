import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt




class conv_Relu_Block(nn.Module):
    def __init__(self):
        super(conv_Relu_Block, self).__init__()
        self.conv=nn.Conv2d(64,64,3,padding=1,bias=False)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.conv(x))
    
class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(conv_Relu_Block,18)
        self.input = nn.Conv2d(3,64,3,padding=1,bias=False)
        self.output = nn.Conv2d(64,3,3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self._initialize_weights()
        
    def make_layer(self, block, num_of_layer):
        layers= []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self,x):
        residual = x 
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out       

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))   
