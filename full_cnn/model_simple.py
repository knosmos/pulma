import torch
import torch.nn as nn

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

class FeatureExtractor(nn.Module):
    def __init__(self, length, n_mels):
        super().__init__()
        self.length = length
        self.n_mels = n_mels

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )

        self.cnn.apply(self.init_weights)
    
    def forward(self, x):
        a = x[:, None, :, :]
        #print(a.shape)
        x = self.cnn(a)
        return x

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

class Baseline(nn.Module):
    def __init__(self, length, n_mels, num_classes=6):
        super().__init__()
        self.length = length
        self.n_mels = n_mels
        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(length, n_mels)
        self.framenet = FrameNet(4 * 128, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        #print(x.shape)
        y = x.transpose(1, 2)
        y = y.flatten(2)
        #print(y.shape)
        outputs = self.framenet(y)
        return outputs