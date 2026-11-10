from .backbones import ClipViTEncoder, ResNet50Encoder
from .clip_text import PromptLearner, TextEncoder, build_clip


def clip_embed_dim(clip_model) -> int:
    proj = clip_model.text_projection
    return proj.out_features if hasattr(proj, "out_features") else proj.shape[1]


def build_image_encoder(cfg, clip_model=None):
    """Return (encoder, pixel_mean, pixel_std) for the configured backbone.
    `clip_model` is required for `clip_vit` and optional for `resnet50`."""
    name = cfg.model.backbone
    if name == "resnet50":
        rn = cfg.model.resnet50
        if clip_model is not None and clip_embed_dim(clip_model) != rn.proj_dim:
            raise ValueError(f"model.resnet50.proj_dim={rn.proj_dim} does not match the CLIP "
                             f"embedding size {clip_embed_dim(clip_model)}")
        return ResNet50Encoder(rn, rn.proj_dim), rn.pixel_mean, rn.pixel_std
    if name == "clip_vit":
        return ClipViTEncoder(clip_model), cfg.clip.pixel_mean, cfg.clip.pixel_std
    raise KeyError(f"unknown backbone '{name}'")


__all__ = ["ClipViTEncoder", "ResNet50Encoder", "PromptLearner", "TextEncoder", "build_clip",
           "build_image_encoder", "clip_embed_dim"]
