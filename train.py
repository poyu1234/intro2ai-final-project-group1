import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from AutoEncoder import AutoEncoder
from tqdm import tqdm

from data import generate_images
from dataset import TrainingDataset, TestDataset

# parameters
num_images = 1000  # number of images to generate
image_size = 224  # size of the images
num_texts = 5  # number of texts to add to each image
add_text = True  # whether to add text to images


# create path for images
original_path = 'dataset/original/'
if not os.path.exists(original_path):
    os.makedirs(original_path)

noisy_path = 'dataset/noisy/'
if not os.path.exists(noisy_path):
    os.makedirs(noisy_path)

result_path = 'dataset/result/'
if not os.path.exists(result_path):
    os.makedirs(result_path)



# generate training dataset
original_images,noisy_images = generate_images(num_images, num_texts, image_size, add_text=add_text)


# load training data
training_dataset = TrainingDataset(original_images, noisy_images)
training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)


# initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)


# model config
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,betas=(0.9, 0.999),weight_decay=1e-6)


# training
epochs = 10
for epoch in range(epochs):  
    model.train() # Set model to training mode
    epoch_loss = 0.0
    progress_bar = tqdm(enumerate(training_loader), total=len(training_loader), desc=f"Epoch {epoch+1}/{epochs}")
    for i, (noisy_batch, original_batch) in progress_bar:
        noisy_batch = noisy_batch.to(device)
        original_batch = original_batch.to(device)
        
        output = model(noisy_batch)
        loss = criterion(output, original_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_epoch_loss = epoch_loss / len(training_loader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")


# generate test dataset
original_images,noisy_images = generate_images(100,5,224, add_text=False)


# save original / noisy images
for idx, original_image in enumerate(original_images):
    original_image.save(os.path.join(original_path, f'original_{idx}.jpg'))

for idx, noisy_image in enumerate(noisy_images):
    noisy_image.save(os.path.join(noisy_path, f'noisy_{idx}.jpg'))


# load test data
test_dataset = TestDataset(noisy_images)
test_loader = DataLoader(test_dataset)


# test model
model.eval()
outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        output = model(batch)
        outputs.append(output.cpu())


# save result images
output = torch.cat(outputs, dim=0)
to_pil = transforms.ToPILImage()
result_images = [to_pil(output[i].cpu().clamp(0,1)) for i in range(output.size(0))]

for idx, result_image in enumerate(result_images):
    result_image.save(os.path.join(result_path, f'result_{idx}.jpg'))