import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import XRayDataset
from src.model import get_model
from src.train import train_model


def entry_point():
    model = get_model()
    
    dataset = XRayDataset(r"..\train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Starting training...")
    train_model(model, dataloader, criterion, optimizer)
    
