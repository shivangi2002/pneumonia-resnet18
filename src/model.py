import torchvision.models as models
import torch.nn as nn

def get_model():
    model = models.resnet18(
        weights = models.ResNet18_Weights.DEFAULT,
        )
    
    for parm in model.parameters():
        parm.requires_grad = False
        
        
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model