import numpy as np
import matplotlib.pyplot as plt

def imshow(images,mean,std):
    img = images.cpu().numpy().transpose((1,2,0))
    #mean = np.array([0.7898755 ,0.86214995, 0.03470522])
    #std = np.array([0.24056949, 0.15234467, 0.14983375])
    mean = np.array(mean)
    std = np.array(std)
    img = std*img +mean
    img = np.clip(img,0,1)

    plt.imshow(img)
    #plt.show()

def reverse_nor(images,mean,std):
    img = images.cpu().numpy().transpose((1,2,0))
    mean = np.array(mean)
    std = np.array(std)
    img = std*img +mean
    img = np.clip(img,0,1)

    return img

def image_normalization_extract(nor_test_data_set):
    meanRGB = [np.mean(x.numpy(),axis=(1,2)) for x,_ in nor_test_data_set]
    stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in nor_test_data_set]

    meanR =np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    stdR = np.mean([m[0] for m in stdRGB])
    stdG = np.mean([m[1] for m in stdRGB])
    stdB = np.mean([m[2] for m in stdRGB])

    print("mean ", meanR, meanG, meanB)
    print("std ", stdR, stdG, stdB)
    nor_mean,nor_std = (meanR, meanG, meanB),(stdR, stdG, stdB)

    return nor_mean,nor_std
