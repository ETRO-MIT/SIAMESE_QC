"""V11 architectures; original state-dict keys, no ImageNet download."""
# Python packages
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self, out_nodes: int):
        super(SiameseNet, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_nodes)

    def forward(self, x: torch.Tensor):
        out = self.resnet(x)
        out = F.normalize(out, p=2, dim=1)

        return out

# Network that estimates the Euclidean distance
class SingleDistanceNet(nn.Module):
    def __init__(self, siamese_out_nodes: int):
        super(SingleDistanceNet, self).__init__()
        # Define the ResNet-50 model
        self.resnet50 = torchvision.models.resnet50(weights=None)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, siamese_out_nodes)

        # Define the regression head
        self.regression = nn.Sequential(
            nn.Linear(in_features=siamese_out_nodes, out_features=1),
            nn.Softplus()
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.resnet50(images)
        output = self.regression(output)

        return output


# Network that classifies patches as correct or incorrect
class ClassificationNet(nn.Module):
    def __init__(self, siamese_out_nodes: int):
        super(ClassificationNet, self).__init__()
        # Define the ResNet-50 model
        self.resnet50 = torchvision.models.resnet50(weights=None)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, siamese_out_nodes)

        # Define the classification head
        self.classification = nn.Sequential(
            nn.Linear(in_features=siamese_out_nodes, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.resnet50(images)
        output = self.classification(output)

        return output
