import segmentation_models_pytorch as smp

import config


def create_single_unet():
    if config.dataset == "sn6":
        input_channels = 4
    else:
        input_channels = 2
    model = smp.Unet(
        encoder_name="resnet18",  # TODO: read about skipconnections usage
        encoder_weights=None,
        decoder_use_batchnorm=False,
        in_channels=input_channels,
        classes=2,
    )
    return model
