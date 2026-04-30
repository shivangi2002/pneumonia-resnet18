import os

import torch
import torch.nn as nn
import csv
from torch.utils.data import DataLoader,random_split

from src.dataset import XRayDataset
from src.model import get_model
from src.train import train_model

def run_model(lr, images_per_batch, num_epochs):
    torch.manual_seed(42)
    model = get_model()
    
    full_train_dataset = XRayDataset(r"..\data\train")
    train_size = int(0.8 * len(full_train_dataset)) 
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
        )
    
    
    test_dataset = XRayDataset(r"..\data\test")
    
    
    train_loader = DataLoader(train_dataset, batch_size=images_per_batch, shuffle=True )
    val_loader = DataLoader(val_dataset, batch_size=images_per_batch, shuffle=False )
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    best_epoch = history["val_loss"].index(min(history["val_loss"]))
    file_exists = os.path.isfile("docs/hyperparameters_tuning_results.csv")
    with open(
        "docs/hyperparameters_tuning_results.csv",
        mode="a",
        newline=""
    ) as file:
        writer = csv.writer(file)
        if not file_exists:
             writer.writerow(
                ["LR", "Batch Size", "Num Epochs","Best Epoch","Train Loss", "Val Loss", "Val Accuracy", "Val Precision", "Val Recall"]
            )
        
        writer.writerow(
            [lr, images_per_batch, num_epochs,best_epoch, history["train_loss"][best_epoch], history["val_loss"][best_epoch], history["val_accuracy"][best_epoch], history["val_precision"][best_epoch], history["val_recall"][best_epoch]])