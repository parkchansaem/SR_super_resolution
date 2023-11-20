import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torchvision.utils import save_image

from util.metrics import psnr,SSIM



def train(model, data_dl,optimizer,device,loss_func):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    for data in tqdm(data_dl):
        image = data[0].to(device)
        label = data[1].to(device)

        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_func(outputs, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1 / optimizer.param_groups[0]["lr"], norm_type=2.0)
        optimizer.step()

        running_loss += loss.item()
        batch_psnr = psnr(label, outputs)
        running_psnr +=batch_psnr
        batch_ssim = SSIM(label, outputs)
        running_ssim +=batch_ssim

    
    final_loss= running_loss/ len(data_dl.sampler)
    final_psnr = running_psnr/int(len(data_dl.sampler)/ data_dl.batch_size)
    final_ssim = running_ssim/int(len(data_dl.sampler)/ data_dl.batch_size)
    return final_loss, final_psnr,final_ssim

def validate(model, data_dl,device,loss_func,epoch):

    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    with torch.no_grad():
        for data in tqdm(data_dl):
            image = data[0].to(device)
            label = data[1].to(device)

            outputs = model(image)
            loss =loss_func(outputs, label)

            running_loss += loss.item()
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr
            batch_ssim = SSIM(label, outputs)
            running_ssim +=batch_ssim

        outputs = outputs.cpu()

        #save_image(outputs, f'./save_img/vdsr/{epoch}.png')
    
    final_loss= running_loss/len(data_dl.sampler)
    final_psnr = running_psnr/ int(len(data_dl.sampler)/data_dl.batch_size)
    final_ssim = running_ssim/ int(len(data_dl.sampler)/ data_dl.batch_size)
  

    return final_loss, final_psnr,final_ssim