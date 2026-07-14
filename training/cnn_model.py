import torch
import torch.nn as nn
import torchvision.models as models


class CropDiseaseModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()


        # MobileNetV2 Transfer Learning

        self.model = models.mobilenet_v2(

            weights="DEFAULT"

        )


        # Freeze Pretrained Layers

        for param in self.model.parameters():

            param.requires_grad = False


        # Replace Classifier

        self.model.classifier[1] = nn.Linear(

            self.model.last_channel,

            num_classes

        )


    def forward(self, x):

        return self.model(x)



# Test Model

if __name__ == "__main__":


    model = CropDiseaseModel(

        num_classes=9

    )


    print(model)