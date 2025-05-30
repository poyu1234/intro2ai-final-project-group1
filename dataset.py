from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import torch

class TrainingDataset(Dataset):
    def __init__(self, original_images, noisy_images):
        self.original_images = original_images
        self.noisy_images = noisy_images

        # Basic transform to tensor and grayscale
        self.base_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.original_images)

    def _augment(self, img):
        # Gentle geometric augmentations
        if random.random() < 0.3:
            angle = random.uniform(-2, 2)
            img = img.rotate(angle, fillcolor=255)
        if random.random() < 0.3:
            img = img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, random.uniform(-2, 2), 0, 1, random.uniform(-2, 2)),
                fillcolor=255
            )
        # Slight contrast enhancement
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(1.1, 1.3))
        # Slight blur (simulates scan/copy)
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.5)))
        return img

    def _normalize(self, tensor):
        # Adaptive histogram equalization for better line contrast
        np_img = (tensor.squeeze().numpy() * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        np_img = clahe.apply(np_img)
        return torch.from_numpy(np_img / 255.0).unsqueeze(0).float()

    def __getitem__(self, index):
        orig_img = self.original_images[index].convert('L')
        noisy_img = self.noisy_images[index].convert('L')

        # Synchronized augmentations
        seed = random.randint(0, 99999)
        random.seed(seed)
        orig_img = self._augment(orig_img)
        random.seed(seed)
        noisy_img = self._augment(noisy_img)

        orig_tensor = self.base_transform(orig_img)
        noisy_tensor = self.base_transform(noisy_img)

        # Normalize for table structure
        orig_tensor = self._normalize(orig_tensor)
        noisy_tensor = self._normalize(noisy_tensor)

        return noisy_tensor, orig_tensor

class TestDataset(Dataset):
    def __init__(self, test_images):
        self.test_images = test_images
        self.base_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def _normalize(self, tensor):
        np_img = (tensor.squeeze().numpy() * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        np_img = clahe.apply(np_img)
        return torch.from_numpy(np_img / 255.0).unsqueeze(0).float()

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, index):
        img = self.test_images[index]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            img = img.convert('L')
        tensor = self.base_transform(img)
        tensor = self._normalize(tensor)
        return tensor