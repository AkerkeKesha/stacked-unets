import segmentation_models_pytorch as smp
import config
import torch.nn as nn
from copy import deepcopy


class StackedUNet(nn.Module):
    def __init__(self, n_levels, base_model):
        super().__init__()
        self.n_levels = n_levels
        self.base_model = base_model
        if n_levels > 0:
            self.models = nn.ModuleList([deepcopy(base_model) for _ in range(n_levels)])
        else:
            self.models = None

    def forward(self, x):
        if self.n_levels > 0:
            out = x
            for i in range(self.n_levels):
                out = self.models[i](out)
            return out
        else:
            return self.base_model(x)


basic_unet = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        decoder_use_batchnorm=False,
        in_channels=4 if config.dataset == "sn6" else 2,
        classes=2,
)

