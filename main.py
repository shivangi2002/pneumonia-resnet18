import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

from src.dataset import XRayDataset
from src.model import get_model
from src.train import train_model
from src.visualize import plot_loss_curve
from src.logger import save_run_summary, save_run_history

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_model(lr, images_per_batch, num_epochs):
    torch.manual_seed(42)
    model_save_path = os.path.join(PROJECT_ROOT, "checkpoints", f"lr{lr}_bs{images_per_batch}_epochs{num_epochs}.pth")    
    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "history"), exist_ok=True)   
    
    model = get_model()
    
    full_train_dataset = XRayDataset(os.path.join(PROJECT_ROOT, "data", "train"))
    train_size = int(0.8 * len(full_train_dataset)) 
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
        )
    
    
    test_dataset = XRayDataset(os.path.join(PROJECT_ROOT, "data", "test"))
    
    
    train_loader = DataLoader(train_dataset, batch_size=images_per_batch, shuffle=True )
    val_loader = DataLoader(val_dataset, batch_size=images_per_batch, shuffle=False )
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,model_save_path)
    best_epoch = history["val_loss"].index(min(history["val_loss"]))
   
    hyperparameters = {
        "lr": lr,
        "images_per_batch": images_per_batch,
        "num_epochs": num_epochs
    }
    csv_path = os.path.join(PROJECT_ROOT, "results", "hyperparameters_tuning_results.csv")
    save_run_summary(hyperparameters, best_epoch, history, csv_path)
    
    plot_save_path = os.path.join(PROJECT_ROOT, "plots", f"lr{lr}_bs{images_per_batch}_epochs{num_epochs}.png")  
    plot_loss_curve(history["train_loss"], history["val_loss"], plot_save_path)
    
    history_save_path = os.path.join(PROJECT_ROOT, "results","history", f"lr{lr}_bs{images_per_batch}_epochs{num_epochs}.json")
    save_run_history(history, history_save_path)