import segmentation_models_pytorch as smp


def create_single_unet():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=2,
    )
    return model
