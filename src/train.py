from torch.utils.data import DataLoader
from dataset import XRayDataset

dataset = XRayDataset(r"..\train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True )
