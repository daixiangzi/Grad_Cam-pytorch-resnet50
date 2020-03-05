from torch.utils.data import Dataset
import torch
from PIL import Image
class MyDataset(Dataset):
    def __init__(self,train_file,transform=None):
            f = open(train_file,'r')
            self.content = f.readlines()
            self.transform = transform
    def __len__(self):
            return len(self.content)
    def __getitem__(self,index):
            temp = self.content[index]
            image_path = temp.strip('\n').split(',')[0]
            label = int(temp.strip('\n').split(',')[1])
            image = Image.open(image_path)#pytorch c*h*w
            image = self.transform(image)
            return image,label
