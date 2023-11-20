import matplotlib.pyplot as plt
from util.normalization import imshow, reverse_nor
from util.metrics import psnr, SSIM

def output_visualization(HR_label,LR_img,output,bicu_img,nor_mean,nor_std):
    # visualization
    plt.figure(figsize=(15,15))
    plt.subplot(1,4,1)
    imshow(HR_label,nor_mean,nor_std)
    #plt.imshow(to_pil_image(img), cmap='gray')
    plt.title('HR_label')
    plt.subplot(1,4,2)
    imshow(LR_img,nor_mean,nor_std)
    #plt.imshow(to_pil_image(output), cmap='gray')
    plt.title('LR_img')
    plt.subplot(1,4,3)
    imshow(output,nor_mean,nor_std)
    #plt.imshow(to_pil_image(label), cmap='gray')
    plt.title('output')
    plt.subplot(1,4,4)
    imshow(bicu_img,nor_mean,nor_std)
    #plt.imshow(to_pil_image(label), cmap='gray')
    plt.title('bicubic')
    plt.show()
    print('PSNR')
    print('output:',psnr(reverse_nor(HR_label,nor_mean,nor_std),reverse_nor(output,nor_mean,nor_std),numpy_v=True))
    # print('PSNR')
    # print('input:',psnr(reverse_nor(label,nor_mean,nor_std),reverse_nor(img,nor_mean,nor_std),numpy_v=True), 'output:',psnr(reverse_nor(label,nor_mean,nor_std),reverse_nor(output,nor_mean,nor_std),numpy_v=True) )
    # print('SSIM')
    # print('input:',SSIM(reverse_nor(label,nor_mean,nor_std),reverse_nor(img,nor_mean,nor_std),numpy_v=True), 'output:',SSIM(reverse_nor(label,nor_mean,nor_std),reverse_nor(output,nor_mean,nor_std),numpy_v=True) )