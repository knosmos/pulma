import torch
import torch.nn as nn
from torchvision import models

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
        # sigmoid for mult-label
        x = torch.sigmoid(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
