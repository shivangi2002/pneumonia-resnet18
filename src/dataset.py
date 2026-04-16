import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class XRayDataset (Dataset):
    def __init__(self,dataset_path):
        self.samples = []
        self.class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
        for class_name in self.class_to_idx.keys():
            class_path = os.path.join(dataset_path, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.samples.append((img_path, self.class_to_idx[class_name]))
    
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
                                                                                                                            
        
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")  
        img = self.transform(img)
        return img, label