import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = self._block(features, features*2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = self._block(features*2, features*4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = nn.Sequential(
            self._block(features*4, features*8, dilation=2),
            nn.Dropout2d(0.5)
        )
        
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, 2, 2)
        self.attn3 = ChannelAttention(features*8)
        self.decoder3 = self._block(features*8, features*4)
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, 2, 2)
        self.attn2 = ChannelAttention(features*4)
        self.decoder2 = self._block(features*4, features*2)
        self.upconv1 = nn.ConvTranspose2d(features*2, features, 2, 2)
        self.attn1 = ChannelAttention(features*2)
        self.decoder1 = self._block(features*2, features)
        self.conv = nn.Conv2d(features, out_channels, 1)
        
    def _block(self, in_channels, features, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, 3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.attn3(dec3)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.attn2(dec2)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.attn1(dec1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))
