import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .dit import TransformerCondition


NUM_CLASSES = 1000


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def model_and_diffusion_joints_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        joint_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res




def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    lambda_pos_loss=1,
    lambda_rcxyz=1,
    lambda_mse=1,
    lambda_vb=1,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_pos_loss=lambda_pos_loss,
        # lambda_rcxyz=lambda_rcxyz,
        lambda_mse=lambda_mse,
        lambda_vb=lambda_vb,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def create_model_condition_and_diffusion(
    image_size,
    in_channels,
    joint_cond,
    class_cond,
    learn_sigma,
    num_channels,
    channel_mult,
    num_heads,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_fp16,
    joint_cond_L=False,
    add_cond=False,
    in_dim=None,
    out_channels=6,
    lambda_pos_loss=1,
    njoints=22, 
    nfeats=6,
    ff_size=1024, 
    num_layers=8,
    activation="gelu", 
    legacy=False, 
    data_rep='rot', 
    dataset='amass', 
    arch='trans_enc', 
    emb_trans_dec=False,
    lambda_rcxyz=1,
    lambda_mse=1,
    lambda_vb=1,
    sigma_small=False,
    batch_size=256, 
    mlp_ratio=4.0,
    pos_enc=False,
    init_weights=False,
):
    model = create_model_condition(
        image_size,
        in_channels,
        num_channels,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        joint_cond=joint_cond,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        dropout=dropout,
        use_fp16=use_fp16,
        cond_dim=(18 if joint_cond_L else 6),
        add_cond=add_cond,
        in_dim=in_dim,
        out_channels=out_channels,
        njoints=njoints,
        nfeats=nfeats,
        ff_size=ff_size,
        num_layers=num_layers,
        activation=activation,
        legacy=legacy,
        data_rep=data_rep,
        dataset=dataset,
        arch=arch,
        emb_trans_dec=emb_trans_dec,
        batch_size=batch_size, 
        mlp_ratio=mlp_ratio,
        pos_enc=pos_enc,
        init_weights=init_weights,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        lambda_pos_loss=lambda_pos_loss,
        lambda_rcxyz=lambda_rcxyz,
        sigma_small=sigma_small,
        lambda_mse=lambda_mse,
        lambda_vb=lambda_vb,
    )
    return model, diffusion


def create_model_condition(
    image_size,
    in_channels,
    num_channels,
    channel_mult="",
    learn_sigma=False,
    joint_cond=False,
    attention_resolutions=-1,
    num_heads=1,
    dropout=0,
    use_fp16=False,
    cond_dim=6,
    add_cond=False,
    in_dim=None,
    out_channels=6,
    njoints=22, 
    nfeats=6,
    ff_size=1024, 
    num_layers=8,
    activation="gelu", 
    legacy=False, 
    data_rep='rot', 
    dataset='amass', 
    arch='trans_enc', 
    emb_trans_dec=False,
    batch_size=256, 
    mlp_ratio=4.0,
    pos_enc=False,
    init_weights=False,
):
    if channel_mult == -1:
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif isinstance(image_size, list):
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(channel_mult)
    # ims = 128
    attention_ds = []
    for res in attention_resolutions:
        attention_ds.append([ims // res for ims in image_size])
    return TransformerCondition(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(out_channels if not learn_sigma else 2 * out_channels),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        joint_cond=joint_cond,
        cond_dim=cond_dim,
        add_cond=add_cond,
        in_dim=in_dim,
        njoints=njoints,
        nfeats=nfeats,
        ff_size=ff_size,
        num_layers=num_layers,
        activation=activation,
        legacy=legacy,
        data_rep=data_rep,
        dataset=dataset,
        arch=arch,
        emb_trans_dec=emb_trans_dec,
        batch_size=batch_size, 
        mlp_ratio=mlp_ratio,
        pos_enc=pos_enc,
        init_weights=init_weights,
    )
