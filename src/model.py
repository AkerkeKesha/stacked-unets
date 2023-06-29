import segmentation_models_pytorch as smp
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,            # model input channels: vv, vh and prev_model_output
            classes=2,                # binary segmentation
        )

    def forward(self, x):
        return self.model(x)


class StackedUNet(nn.Module):
    def __init__(self, n_levels):
        super().__init__()
        self.n_levels = n_levels
        self.models = nn.ModuleList([smp.Unet(encoder_name="resnet18",
                                              encoder_weights=None,
                                              in_channels=3,
                                              classes=2,) for _ in range(n_levels)])

    def forward(self, x):
        out = x
        for i in range(self.n_levels):
            out = self.models[i](out)
        return out





