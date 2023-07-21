'''
U-Net 2D CNN structure for preprocessing of 15-second lung signal,
in order to maintain original dimensions before feeding to MLP/LSTM/GRU.
'''

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.enc_blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i+1])
            for i in range(len(channels)-1)
        ])
    
    def forward(self, x):
        outputs = []
        for block in self.enc_blocks:
            x = block(x)
            outputs.append(x)
        return outputs # return list of outputs from each block (include intermediate steps for decoder)

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Convolutional blocks
        self.dec_blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i+1])
            for i in range(len(channels)-1)
        ])

        ## Upsampling blocks
        self.upsample_blocks = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2)
            for i in range(len(channels)-1)
        ])

    def forward(self, x, enc_outputs):
        for i in range(len(self.channels)-1):
            x = self.upsample_blocks[i](x)
            # Concatenate encoder output with decoder output
            x = torch.cat([x, enc_outputs[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x

class FrameNet(nn.Module):
    def __init__(self, input_size, num_classes=6):
        super(FrameNet, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Simple MLP
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self.model.apply(self.init_weights)
        
    def forward(self, x):
        x = self.model(x)
        # sigmoid for multi-label
        x = torch.sigmoid(x)
        return x
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

class UNet(nn.Module):
    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

    def forward(self, x):
        enc_outputs = self.encoder(x)
        x = self.decoder(enc_outputs[::-1][0], enc_outputs[::-1][1:])
        return x

class Baseline(nn.Module):
    def __init__(self, input_size, num_classes=6):
        super(FullCNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.unet = UNet([])
        self.framenet = FrameNet(input_size, num_classes)

    def forward(self, x):
        x = self.unet(x)
        outputs = []
        for i in x:
            outputs.append(self.framenet(i))
        return outputs