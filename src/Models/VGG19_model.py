import torch
import torch.nn as nn

NUM_CLASSES = 4

class VGG19(nn.Module):
  def __init__(self, original_vgg19_model):
    super(VGG19, self).__init__()

    self.original_vgg19_model = original_vgg19_model
    self.features = self.original_vgg19_model.features
    self.classifier = self.original_vgg19_model.classifier
    self.switch_indices = [0] * len(self.features)

    for layer in self.features:
      if isinstance(layer, nn.MaxPool2d):
        layer.return_indices = True
      if isinstance(layer, nn.ReLU):
        layer.inplace = True

    in_features = self.classifier[-1].in_features
    final_classifier_layer = len(self.classifier) - 1

    self.classifier[final_classifier_layer] = nn.Linear(4096, NUM_CLASSES)

  def forward(self, batch: torch.Tensor):
    for i, layer in enumerate(self.features):
      if isinstance(layer, nn.MaxPool2d):
        batch, indices = layer(batch)
        self.switch_indices[i] = indices
      else:
        batch = layer(batch)

    batch = batch.view(batch.size(0), -1)
    batch_probabilities = self.classifier(batch)

    return batch_probabilities