import torch
from src.eval import validate_model
def train_model(model,train_loader,validation_loader,criterion,optimizer,num_epochs):
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],  
        "val_precision": [],
        "val_recall": []
    }
    for epoch in range(num_epochs):
        
        total_train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)    
        
        history["train_loss"].append(avg_train_loss)
        
        val_loss, val_accuracy, val_precision, val_recall = validate_model(model, validation_loader, criterion)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)             
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall) 

        print(f"Epoch {epoch:d} | train Loss: {avg_train_loss:8.4f}  | val_loss: {val_loss:8.4f} | val_accuracy: {val_accuracy:8.4f} | val_precision: {val_precision:8.4f} | val_recall: {val_recall:8.4f}")                           
                                                    
    return history