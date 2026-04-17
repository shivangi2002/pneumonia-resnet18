from torch.utils.data import DataLoader
from dataset import XRayDataset
import torch.nn as nn

dataset = XRayDataset(r"..\train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True )

criterion = nn.CrossEntropyLoss()