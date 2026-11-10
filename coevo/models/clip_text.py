"""CLIP construction and the prompt-learning text side of CSA.

`PromptLearner` and `TextEncoder` are adapted from CLIP-ReID
(model/make_model_clipreid.py): one set of learnable context tokens per identity inside
the fixed template "A photo of a X ... X person.". Differences: the number of tokens L
is a parameter (CLIP-ReID hard-codes 4) and the CLIP implementation is open_clip."""
from __future__ import annotations

import open_clip
import torch
from torch import nn


def build_clip(clip_cfg, image_size: tuple[int, int]):
    """Return (clip_model, tokenizer). With `clip_cfg.tiny` set, a small randomly
    initialised CLIP is built instead (smoke tests, no download)."""
    if clip_cfg.tiny is not None:
        t = clip_cfg.tiny
        model = open_clip.CLIP(
            embed_dim=t.embed_dim,
            vision_cfg=open_clip.CLIPVisionCfg(layers=t.layers, width=t.width, head_width=t.width // t.heads,
                                               patch_size=t.patch_size, image_size=image_size),
            text_cfg=open_clip.CLIPTextCfg(context_length=77, vocab_size=49408, width=t.width,
                                           heads=t.heads, layers=t.layers),
            quick_gelu=True,
        )
        return model, open_clip.get_tokenizer("ViT-B-16")
    model = open_clip.create_model(clip_cfg.arch, pretrained=clip_cfg.pretrained,
                                   force_image_size=image_size,
                                   force_quick_gelu=clip_cfg.pretrained == "openai")
    return model, open_clip.get_tokenizer(clip_cfg.arch)


class TextEncoder(nn.Module):
    """CLIP text tower applied to prompt *embeddings* instead of token ids."""

    def __init__(self, clip_model):
        super().__init__()
        self.clip = [clip_model]  # held in a list: not registered, never part of a state dict

    def forward(self, prompts: torch.Tensor, eot_index: int) -> torch.Tensor:
        clip = self.clip[0]
        x = prompts + clip.positional_embedding.to(prompts.dtype)
        x = clip.transformer(x, attn_mask=clip.attn_mask)
        x = clip.ln_final(x)
        x = x[:, eot_index]
        proj = clip.text_projection
        return proj(x) if isinstance(proj, nn.Module) else x @ proj


class PromptLearner(nn.Module):
    def __init__(self, num_ids: int, num_tokens: int, clip_model, tokenizer):
        super().__init__()
        template = "A photo of a " + " ".join(["X"] * num_tokens) + " person."
        tokens = tokenizer([template])  # [1, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokens.to(clip_model.token_embedding.weight.device))
        prefix_len = 1 + 4  # <start> + "a photo of a"
        ctx_dim = embedding.shape[-1]
        self.cls_ctx = nn.Parameter(torch.empty(num_ids, num_tokens, ctx_dim).normal_(std=0.02))
        self.register_buffer("token_prefix", embedding[:, :prefix_len].clone())
        self.register_buffer("token_suffix", embedding[:, prefix_len + num_tokens:].clone())
        self.eot_index = int(tokens[0].argmax())

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        n = labels.shape[0]
        return torch.cat([self.token_prefix.expand(n, -1, -1), self.cls_ctx[labels],
                          self.token_suffix.expand(n, -1, -1)], dim=1)
