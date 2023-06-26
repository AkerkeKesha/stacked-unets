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





