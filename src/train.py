import torch

def train_model(model,dataloader,criterion,optimizer,num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in dataloader:
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)    
        print(f"Epoch: {epoch}, Loss: {avg_loss}")
    return avg_loss 