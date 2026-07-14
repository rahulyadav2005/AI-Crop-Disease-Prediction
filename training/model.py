import torch
import torch.nn as nn
import torchvision.models as models


class CropDiseaseModel(nn.Module):

    def __init__(self, num_classes=9):

        super().__init__()

        self.model = models.mobilenet_v2(weights="DEFAULT")

        self.model.classifier[1] = nn.Linear(
            self.model.last_channel,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = CropDiseaseModel()

    print(model)