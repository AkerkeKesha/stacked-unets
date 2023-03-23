import segmentation_models_pytorch as smp


def create_single_unet():
    model = smp.Unet(
        # TODO: read about skipconnections usage
        encoder_name="resnet18",
        encoder_weights=None,
        decoder_use_batchnorm=False, # TODO: 2 images, do not train on RGB image
        in_channels=3,  # TODO: 2 images, do not train on RGB image
        classes=2,  # TODO water(contains flood) and land as labels first
    )
    return model
