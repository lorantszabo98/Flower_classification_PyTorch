import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, image_size):
        super(SimpleCNN, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(3, self.image_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the output size of the last convolutional layer
        conv_out_size = image_size * (image_size // 2) * (image_size // 2)

        self.fc1 = nn.Linear(conv_out_size, 9)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x