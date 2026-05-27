import torchvision.models as models
import torch.nn as nn

def get_model(fine_tune_layers = 0):
    model = models.resnet18(
        weights = models.ResNet18_Weights.DEFAULT,
        )
    
    for param in model.parameters():
        param.requires_grad = False
    
    if fine_tune_layers >= 1:
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    if fine_tune_layers >= 2:
        for param in model.layer3.parameters():
            param.requires_grad = True
                
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
