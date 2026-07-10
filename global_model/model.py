import torch
import torch.nn as nn

class GlobalAutoEncoder(nn.Module):
    def __init__(self):
        super(GlobalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.shape[-2:] != (28, 28):
            x = torch.nn.functional.interpolate(x, size=(28, 28))
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)