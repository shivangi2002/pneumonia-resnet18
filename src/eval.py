import torch

import src.dataset as dataset

def validate_model(model, dataloader, criterion):
    model.eval()
    POSITIVE_CLASS = 1
    
    total_loss = 0
    correct = 0
    total = 0
    
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_positive += ((predicted == POSITIVE_CLASS) & (labels == POSITIVE_CLASS)).sum().item()
            false_positive += ((predicted == POSITIVE_CLASS) & (labels != POSITIVE_CLASS)).sum().item()
            false_negative += ((predicted != POSITIVE_CLASS) & (labels == POSITIVE_CLASS)).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return avg_loss, accuracy, precision, recall