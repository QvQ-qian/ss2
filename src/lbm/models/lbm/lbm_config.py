from typing import List, Literal, Optional, Tuple

from pydantic.dataclasses import dataclass

from ..base import ModelConfig


@dataclass
class LBMConfig(ModelConfig):
    """This is the Config for LBM Model class which defines all the useful parameters to be used in the model.

    Args:

        source_key (str):
            Key for the source image. Defaults to "source_image"

        target_key (str):
            Key for the target image. Defaults to "target_image"

        mask_key (Optional[str]):
            Key for the mask showing the valid pixels. Defaults to None

        latent_loss_type (str):
            Loss type to use. Defaults to "l2". Choices are "l2", "l1"

        pixel_loss_type (str):
            Pixel loss type to use. Defaults to "l2". Choices are "l2", "l1", "lpips"

        pixel_loss_max_size (int):
            Maximum size of the image for pixel loss.
            The image will be cropped to this size to reduce decoding computation cost. Defaults to 512

        pixel_loss_weight (float):
            Weight of the pixel loss. Defaults to 0.0

        timestep_sampling (str):
            Timestep sampling to use. Defaults to "uniform". Choices are "uniform"

        input_key (str):
            Key for the input. Defaults to "image"

        controlnet_input_key (str):
            Key for the controlnet conditioning. Defaults to "controlnet_conditioning"

        adapter_input_key (str):
            Key for the adapter conditioning. Defaults to "adapter_conditioning"

        ucg_keys (Optional[List[str]]):
            List of keys for which we enforce zero_conditioning during Classifier-free guidance. Defaults to None

        prediction_type (str):
            Type of prediction to use. Defaults to "epsilon". Choices are "epsilon", "v_prediction", "flow

        logit_mean (Optional[float]):
            Mean of the logit for the log normal distribution. Defaults to 0.0

        logit_std (Optional[float]):
            Standard deviation of the logit for the log normal distribution. Defaults to 1.0

        guidance_scale (Optional[float]):
            The guidance scale. Useful for finetunning guidance distilled diffusion models. Defaults to None

        selected_timesteps (Optional[List[float]]):
            List of selected timesteps to be sampled from if using `custom_timesteps` timestep sampling. Defaults to None

        prob (Optional[List[float]]):
            List of probabilities for the selected timesteps if using `custom_timesteps` timestep sampling. Defaults to None
    """

    source_key: str = "source_image"
    target_key: str = "target_image"
    mask_key: Optional[str] = None
    latent_loss_weight: float = 1.0
    latent_loss_type: Literal["l2", "l1"] = "l2"
    pixel_loss_type: Literal["l2", "l1", "lpips", "dists", "ea_dists"] = "l2"
    pixel_loss_max_size: int = 512
    pixel_loss_weight: float = 0.0
    # EA-DISTS perceptual loss
    ea_dists_edge_weight: float = 1.0
    ea_dists_use_edge: bool = True
    ea_dists_edge_to_rgb: bool = True
    ea_dists_edge_normalize: bool = True
    ea_dists_resize_to: Optional[int] = None
    # Local face-part edge loss
    local_edge_loss_weight: float = 0.0
    local_edge_parts: Optional[List[str]] = None
    local_edge_dilate_kernel: int = 7
    local_edge_exclude_labels: Optional[List[int]] = None
    local_edge_exclude_dilate_kernel: int = 7
    # Bridge-aware MAAM skip refinement
    use_bridge_maam: bool = False
    bridge_maam_mode: Literal["residual", "direct"] = "residual"
    bridge_maam_levels: Optional[List[str]] = None
    bridge_maam_attn_type: Literal["scsa", "cbam", "none"] = "scsa"
    bridge_maam_alpha_init: float = 0.01
    bridge_maam_attn_bias_init: float = 2.0
    bridge_maam_use_timestep: bool = True
    bridge_maam_zero_init_timestep: bool = True
    bridge_maam_scsa_groups: int = 4
    bridge_maam_scsa_kernels: Optional[List[int]] = None
    bridge_maam_scsa_pool_size: int = 7
    # ArcFace identity loss
    id_loss_weight: float = 0.0
    id_loss_model_path: Optional[str] = None
    id_loss_crop: bool = True
    timestep_sampling: Literal["uniform", "log_normal", "custom_timesteps"] = "uniform"
    logit_mean: Optional[float] = 0.0
    logit_std: Optional[float] = 1.0
    selected_timesteps: Optional[List[float]] = None
    prob: Optional[List[float]] = None
    bridge_noise_sigma: float = 0.001
    # Face parsing adapter
    use_face_adapter: bool = False
    parse_key: str = "parse"
    parse_num_classes: int = 19
    parse_adapter_scale: float = 1.0
    parse_adapter_condition_dropout: float = 0.0
    parse_adapter_include_mid: bool = True
    parse_adapter_zero_init: bool = True
    parse_adapter_use_scale_gates: bool = True
    parse_adapter_gate_init: float = 1.0

    # sketch condition adapter
    use_sketch_face_adapter: bool = False
    sketch_key: Optional[str] = None
    sketch_in_channels: int = 3

    # reserved for future coarse face condition
    coarse_face_key: Optional[str] = None
    use_coarse_face_adapter: bool = False
    coarse_in_channels: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.timestep_sampling == "log_normal":
            assert isinstance(self.logit_mean, float) and isinstance(
                self.logit_std, float
            ), "logit_mean and logit_std should be float for log_normal timestep sampling"

        if self.timestep_sampling == "custom_timesteps":
            assert isinstance(self.selected_timesteps, list) and isinstance(
                self.prob, list
            ), "timesteps and prob should be list for custom_timesteps timestep sampling"
            assert len(self.selected_timesteps) == len(
                self.prob
            ), "timesteps and prob should be of same length for custom_timesteps timestep sampling"
            assert (
                sum(self.prob) == 1
            ), "prob should sum to 1 for custom_timesteps timestep sampling"
