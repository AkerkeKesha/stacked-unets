import segmentation_models_pytorch as smp


def create_single_unet():
    model = smp.Unet(
        encoder_name="resnet18",  # TODO: read about skipconnections usage
        encoder_weights=None,
        decoder_use_batchnorm=False,
        in_channels=1,  # TODO: grayscale 1 channel, RGB 3 channels
        classes=2,  # TODO water(contains flood) and land as labels
    )
    return model
