import torchvision.models as models
import torch.nn as nn

def get_model(fine_tune = False):
    model = models.resnet18(
        weights = models.ResNet18_Weights.DEFAULT,
        )
    if fine_tune:
        for name,parm in model.named_parameters():
            if "layer4" not in name:   
                parm.requires_grad = False
        
    else:
        for param in model.parameters():
            param.requires_grad = False
                
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
