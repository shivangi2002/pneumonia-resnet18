import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import XRayDataset,TransformedSubset
from src.model import get_model
from src.eval import validate_model


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
def run_test_eval(checkpoint_path):
    
    test_dataset_path = os.path.join(PROJECT_ROOT, "data", "test")
    test_dataset_i = XRayDataset(test_dataset_path)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = TransformedSubset(test_dataset_i, transform=test_transform)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    model = get_model()
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    avg_loss, accuracy, precision, recall = validate_model(model, test_loader, criterion)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")         
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    