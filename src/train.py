import torch
from src.eval import validate_model
def train_model(model,train_loader,validation_loader,criterion,optimizer,num_epochs, model_save_path,patience=5, min_delta=0.001):
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],  
        "val_precision": [],
        "val_recall": [],
        "stop_epoch": None
    }
    best_val_loss = float('inf')

    epochs_without_improvement = 0
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
        
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path) 
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1 
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch:d} due to no improvement in validation loss for {patience} consecutive epochs.")
                history["stop_epoch"] = epoch
                break   
             
        history["stop_epoch"] = epoch                                    
    return history