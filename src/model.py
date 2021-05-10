import torch.nn as nn
from catalyst.contrib import registry
from torchvision.models import resnext50_32x4d, resnext101_32x8d, densenet121, densenet201, resnet50
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding


out_neurons = 3


@registry.Model
class PneumoniaNet(nn.Module):
    def __init__(self, encoder_name, pretrained=True):
        super().__init__()
        if 'efficientnet' in encoder_name:
            self.model = EfficientNet.from_name(encoder_name)
            self.model.set_swish(False)
            self.model._conv_stem = Conv2dStaticSamePadding(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=224)
            n_features = self.model._fc.in_features
            self.model._fc = nn.Linear(n_features, out_neurons)

        elif 'resnext' in encoder_name:
            if encoder_name == 'resnext50_32x4d':
                self.model = resnext50_32x4d(pretrained=pretrained)
            elif encoder_name == 'resnext101_32x8d':
                self.model = resnext101_32x8d(pretrained=pretrained)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features=n_features, out_features=out_neurons, bias=True)
            )
        elif 'densenet' in encoder_name:
            if encoder_name == 'densenet121':
                self.model = densenet121(pretrained=pretrained)
            elif encoder_name == 'densenet201':
                self.model = densenet201(pretrained=pretrained)
            self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=n_features, out_features=256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.8),
                nn.Linear(in_features=256, out_features=out_neurons)
            )
        elif 'resnet' in encoder_name:
            self.model = resnet50(pretrained=pretrained)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=n_features, out_features=out_neurons)

    def forward(self, features):
        x = self.model(features)

        return x
