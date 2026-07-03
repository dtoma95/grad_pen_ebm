import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_improved2_attpool2.unet import UNetModel

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.1,
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    channel_mult,
    length_time,
):
    
    # print("CHANNELS:", channel_mult)
    attention_ds = []
    if attention_resolutions !='':
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res)) # 

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        image_size=image_size,
        length_time=length_time
    )


def create_model_wrapper_big(
    image_size=64,
    length_time=18,
    num_channels=128,
    num_res_blocks=2,
    num_heads=2,
    num_heads_upsample=-1,
    attention_resolutions="32,16,8", #"16,8",
    dropout=0.0,
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        channel_mult=(1,2,2,2,4),
        length_time=length_time
    )
    return model

def create_model_wrapper_cone(
    image_size=64,
    length_time=18,
    num_channels=128,
    num_res_blocks=2,
    num_heads=2,
    num_heads_upsample=-1,
    attention_resolutions="32,16,8", #"16,8",
    dropout=0.0,
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        channel_mult=(1,2,3,4),
        length_time=length_time
    )
    return model


def create_model_wrapper_small(
    image_size=32,
    length_time=18,
    num_channels=128,
    num_res_blocks=2,
    num_heads=2,
    num_heads_upsample=-1,
    attention_resolutions="16,8", #"16,8",
    dropout=0.0,
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        channel_mult=(1,2,2,2),
        length_time=length_time
    )
    return model

def create_model_wrapper_small2(
    image_size=32,
    length_time=18,
    num_channels=128,
    num_res_blocks=2,
    num_heads=2,
    num_heads_upsample=-1,
    attention_resolutions="16,8", #"16,8",
    dropout=0.0,
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        channel_mult=(1,2,2,1),
        length_time=length_time
    )
    return model