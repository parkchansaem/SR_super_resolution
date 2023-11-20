import numpy as np
import math

def psnr(label, outputs, max_val=1, numpy_v=False):
    if numpy_v == True:
        label=label
        outputs = outputs
    else:
        label=label.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()
    
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0 :
        return 100
    else:
        psnr = 20* math.log10(max_val/rmse)
        return psnr

def SSIM(x,y,numpy_v = False):
    if numpy_v ==True:
        y = y
        x = x
    else:
        y = y.cpu().detach().numpy()
        x = x.cpu().detach().numpy()

    def mean(img):
        return np.mean(img)
    
    def sigma(img):
        return np.std(img)
    
    def cov(img,img2):
        img_ = np.array(img[:,:], dtype=np.float64)
        img2_ = np.array(img2[:,:],dtype=np.float64)

        return np.mean(img * img2) - mean(img)*mean(img2)
    
    K1 = 0.01
    K2 = 0.03
    L = 255 #pixel range

    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    C3 = C2/2

    L = ( 2 * mean(x) * mean(y) + C1 ) / ( mean(x)**2 + mean(y)**2 + C1 )
    C = ( 2 * sigma(x) * sigma(y) + C2 )/ ( sigma(x)**2 + sigma(y)**2 +C2)
    S = (cov(x,y) + C3) / (sigma(x) * sigma(y) + C3 )

    return L * C * S