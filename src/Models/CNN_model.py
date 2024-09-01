import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )

        self.switch_indices = [0] * len(self.features)

        self.classifier = nn.Sequential(
            nn.Linear(16 * 28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # Image classes = 4
        )

    def forward(self, batch):
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                batch, indices = layer(batch)
                self.switch_indices[i] = indices
            else:
                batch = layer(batch)

        batch = batch.view(batch.size(0), -1)
        batch_probabilities = self.classifier(batch)

        return batch_probabilities
