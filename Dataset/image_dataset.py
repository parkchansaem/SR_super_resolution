from torch.utils.data import Dataset

class imageDataset(Dataset):
    def __init__(self, image_data, labels,bicubic_Labels):
        self.iamge_data = image_data
        self.labels = labels
        self.bicubic_Labels = bicubic_Labels
    
    def __len__(self):
        return (len(self.iamge_data))
    
    def __getitem__(self,index):
        image = self.iamge_data[index]
        label = self.labels[index]
        bicubic_Labels = self.bicubic_Labels[index]
        return image[0], label[0],bicubic_Labels[0]  # class drop