import timm
import torch.nn as nn
# from pprint import pprint


# pprint(timm.list_models(pretrained=True))

class CNNModel(nn.Module):
    def __init__(self, model_arc='xception', num_classes=1):
        super().__init__()
        self.net = timm.create_model(model_arc, pretrained=True, num_classes=num_classes, in_chans=1)
    
    def forward(self, x):
        x = self.net(x)

        return x