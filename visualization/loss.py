import matplotlib.pyplot as plt

def Loss_and_metrics_visualization(train_loss,train_psnr,train_ssim,val_loss,val_psnr,val_ssim):
    plt.figure(figsize= (5,5))
    plt.plot(train_loss, color='orange', label= 'train_loss')
    plt.plot(val_loss, color= 'red', label='val_loss')
    plt.xlabel('Eplochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(5,5))
    plt.plot(train_psnr, color= 'green', label='train_psnr')
    plt.plot(val_psnr, color='blue', label='val_psnr')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR(db)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(5,5))
    plt.plot(train_ssim, color= 'black', label='train_ssim')
    plt.plot(val_ssim, color='gray', label='val_ssim')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.show()