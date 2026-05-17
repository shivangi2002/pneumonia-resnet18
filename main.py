import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torchvision import transforms

from src.dataset import XRayDataset, TransformedSubset
from src.model import get_model
from src.train import train_model
from src.visualize import plot_loss_curve
from src.logger import save_run_summary, save_run_history

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_model(lr, images_per_batch, num_epochs, augment=False, patience = 5, class_weights = None, fine_tune_layers= 0):
    torch.manual_seed(42)
    aug_suffix = "_aug" if augment else ""
    cw_suffix = f"_w{int(class_weights[0])}" if class_weights and class_weights[0] != 1.0 else ""
    ft_suffix = f"_ft{fine_tune_layers}" if fine_tune_layers > 0 else ""
    suffix = aug_suffix + cw_suffix+ ft_suffix
    model_save_path = os.path.join(PROJECT_ROOT, "checkpoints", f"lr{lr}_bs{images_per_batch}_epochs{num_epochs}{suffix}.pth")    
    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "history"), exist_ok=True)   
    
    model = get_model(fine_tune_layers=fine_tune_layers)
    
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = val_transform 

    
    full_dataset = XRayDataset(os.path.join(PROJECT_ROOT, "data", "train"))
    train_size = int(0.8 * len(full_dataset)) 
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size]
        )
    
    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset = TransformedSubset(val_subset, val_transform)
    
   
    
    train_loader = DataLoader(train_dataset, batch_size=images_per_batch, shuffle=True )
    val_loader = DataLoader(val_dataset, batch_size=images_per_batch, shuffle=False )
 
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,model_save_path,patience)
    best_epoch = history["val_loss"].index(min(history["val_loss"]))
   
    hyperparameters = {
        "lr": lr,
        "images_per_batch": images_per_batch,
        "num_epochs": num_epochs
    }
    csv_path = os.path.join(PROJECT_ROOT, "results", "hyperparameters_tuning_results.csv")
    save_run_summary(hyperparameters, augment, class_weights, fine_tune_layers, best_epoch, history, csv_path)
    plot_save_path = os.path.join(PROJECT_ROOT, "plots", f"lr{lr}_bs{images_per_batch}_epochs{num_epochs}{suffix}.png")  
    plot_loss_curve(history["train_loss"], history["val_loss"], plot_save_path)
    
    history_save_path = os.path.join(PROJECT_ROOT, "results","history", f"lr{lr}_bs{images_per_batch}_epochs{num_epochs}{suffix}.json")
    save_run_history(history, history_save_path)