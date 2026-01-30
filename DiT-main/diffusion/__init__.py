# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from . import gaussian_diffusion_family as gdf
from . import gaussian_diffusion_all as gdA
from .respace import SpacedDiffusion, space_timesteps, SpacedDiffusionFamily, SpacedDiffusionAll


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
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
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )

def create_diffusion_family(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=False,
    rescale_learned_sigmas=False,
    diffusion_steps=200
):
    betas = gdf.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gdf.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gdf.LossType.RESCALED_MSE
    else:
        loss_type = gdf.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusionFamily(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gdf.ModelMeanType.EPSILON if not predict_xstart else gdf.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gdf.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gdf.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gdf.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )

def create_diffusion_all(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=False,
    rescale_learned_sigmas=False,
    diffusion_steps=200
):
    betas = gdA.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gdA.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gdA.LossType.RESCALED_MSE
    else:
        loss_type = gdA.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusionAll(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gdA.ModelMeanType.EPSILON if not predict_xstart else gdA.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gdA.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gdA.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gdA.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )
