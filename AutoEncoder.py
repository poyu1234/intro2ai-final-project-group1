import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    def forward(self, x):
        return F.relu(x + self.block(x))

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()


        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        

        # decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32)
        )
        self.final = nn.Conv2d(32, 1, 3, padding=1)
        
        # skip connections
        self.skip1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        
        # decoder
        d1 = self.dec1(x2) + self.skip2(x2)
        d2 = self.dec2(d1) + self.skip1(x1)
        return self.sigmoid(self.final(d2))
