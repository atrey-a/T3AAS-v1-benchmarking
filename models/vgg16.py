import torch.nn as nn
from torchvision import models


def VGG16(num_classes):
    num_classes = num_classes
    model = models.vgg16(pretrained=True)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    return model
