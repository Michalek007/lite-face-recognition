import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteFaceCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(LiteFaceCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 3x3 conv, output: 16xHxW
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 3x3 conv, output: 32xHxW
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 3x3 conv, output: 64xHxW
        self.bn3 = nn.BatchNorm2d(64)

        # Adaptive Pooling to ensure output is compatible with dense layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # Assuming input image size is 112x112 and downsampling
        self.fc2 = nn.Linear(256, embedding_dim)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    @property
    def name(self):
        return ''

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Downsample by 2

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Downsample by 2

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Downsample by 2

        # Adaptive pooling to reshape output to 7x7
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  # Final embedding layer

        return x


if __name__ == '__main__':
    model = LiteFaceCNN()
    print(model.name)
    print(model)
    model.eval()
    if 1:
        torch_input = torch.randn(1, 3, 112, 112)
        torch.onnx.export(model, torch_input, "lite_face_cnn.onnx")
