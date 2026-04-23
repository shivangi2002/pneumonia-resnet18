import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

from src.dataset import XRayDataset
from src.model import get_model
from src.train import train_model
from src.eval import validate_model

def entry_point():
    model = get_model()
    
    full_train_dataset = XRayDataset(r"..\data\train")
    train_size = int(0.8 * len(full_train_dataset)) 
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
        )
    
    
    test_dataset = XRayDataset(r"..\data\test")
    
    
    
    
    
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer)
    
    val_loss, val_accuracy, val_precision, val_recall = validate_model(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation Precision: {val_precision}")
    print(f"Validation Recall: {val_recall}")