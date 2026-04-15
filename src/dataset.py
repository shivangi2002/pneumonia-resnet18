from torchvision import models
import torch.nn as nn

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

def freezing_model(model):
    print("Before freezing:")
   
    for param in model.parameters():
        print( param.shape)