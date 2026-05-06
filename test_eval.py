import os
import torch
from torch.utils.data import DataLoader

from src.dataset import XRayDataset
from src.model import get_model
from src.eval import validate_model


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
def run_test_eval(checkpoint_path):
    
    test_dataset_path = os.path.join(PROJECT_ROOT, "data", "test")
    test_dataset = XRayDataset(test_dataset_path)
    
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

    