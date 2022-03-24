from torch import nn


class ModelArchitecture(nn.Module):
    """
    Model architecture with forward method
    """
    def __init__(self):
        super(ModelArchitecture, self).__init__()
        self.super_layer = nn.Conv2d(3, 3, (3, 3))

    def forward(self, x):
        return self.super_layer(x)
