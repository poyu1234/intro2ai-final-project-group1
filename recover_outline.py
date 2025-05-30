import torch
from AutoEncoder import AutoEncoder
from dataset import TestDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

def recover_outline(noisy_outline_image, model_path='model/AutoEncoder.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    
    test_dataset = TestDataset([noisy_outline_image])
    test_loader = DataLoader(test_dataset)
    
    # Fix: Load model with proper device mapping
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        # Map CUDA tensors to CPU when loading on CPU-only machine
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            outputs.append(output.cpu())

    output = torch.cat(outputs, dim=0)
    to_pil = transforms.ToPILImage()
    result_image = [to_pil(output[i].cpu().clamp(0,1)) for i in range(output.size(0))]
    return np.array(result_image[0])


