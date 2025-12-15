from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizationResult:
    quantized: torch.Tensor
    hard: torch.Tensor
    indices: torch.Tensor
    likelihoods: torch.Tensor
    log_likelihoods: torch.Tensor
    aux: Dict[str, torch.Tensor]


def _flatten_z(z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    if z.dim() != 4:
        raise ValueError("Expected tensor with shape [N, C, H, W]")
    n, c, h, w = z.shape
    z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)
    return z_flat, (n, c, h, w)


def _unflatten_z(z_flat: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
    n, c, h, w = shape
    return z_flat.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()


class DifferentiableVectorQuantizer(nn.Module):
    """Differentiable vector quantizer implementing DiVeQ and SF-DiVeQ behaviours."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        sigma2: float = 1e-3,
        prob_decay: float = 0.99,
        warmup_iters: int = 0,
        init_samples_per_code: int = 40,
        init_cache_batches: int = 50,
        replace_schedule: Optional[List[Tuple[int, int, int]]] = None,
        replace_threshold: float = 0.01,
        use_replacement: bool = True,
        variant: str = "original",
    ) -> None:
        super().__init__()
        if num_codes < 2:
            raise ValueError("Codebook size must be at least 2.")
        if variant not in {"original", "detach"}:
            raise ValueError(f"Unknown DiVeQ variant: {variant}")
        self.dim = dim
        self.num_codes = num_codes
        self.sigma = math.sqrt(max(sigma2, 1e-12))
        self.prob_decay = prob_decay
        self.warmup_iters = warmup_iters
        self.init_samples_per_code = max(1, init_samples_per_code)
        self.replace_schedule = replace_schedule or []
        self.replace_threshold = replace_threshold
        self.use_replacement = use_replacement
        self.variant = variant
        self.detach_variant = variant == "detach"

        embed = torch.randn(num_codes, dim)
        self.codebook = nn.Parameter(embed)

        probs = torch.full((num_codes,), 1.0 / num_codes)
        self.register_buffer("ema_probs", probs.clone())
        self.register_buffer("usage_acc", torch.zeros(num_codes))
        self.register_buffer("_counts_since_replace", torch.zeros(num_codes))

        self._initialized = False
        self._init_cache: deque[torch.Tensor] = deque(maxlen=init_cache_batches)
        self._cached_latents = 0
        self._last_replacement_iter = -1

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"num_codes={self.num_codes}, "
            f"sigma2={self.sigma ** 2:.2e}, "
            f"warmup_iters={self.warmup_iters}"
        )

    def forward(self, z: torch.Tensor, iteration: Optional[int] = None) -> QuantizationResult:
        z_flat, shape = _flatten_z(z)
        if self.training and iteration is not None and iteration < self.warmup_iters:
            self._cache_latents(z_flat)
            return self._warmup_passthrough(z, shape)

        if not self._initialized:
            self._initialize_codebook()

        indices = self._nearest_code(z_flat)
        hard = F.embedding(indices, self.codebook)
        approx = self._differentiable_projection(z_flat, indices, hard)
        quantized_flat = approx + (hard - approx).detach()
        quantized = _unflatten_z(quantized_flat, shape)
        hard_spatial = _unflatten_z(hard, shape)

        probs = self._lookup_probs(indices)
        log_probs = torch.log(probs.clamp(min=1e-9))
        likelihoods = self._reshape_probs(probs, shape)
        log_likelihoods = self._reshape_probs(log_probs, shape)
        perplexity = self._perplexity()

        if self.training:
            self._update_usage(indices)
            if self.use_replacement:
                self._maybe_replace_codewords(iteration)

        return QuantizationResult(
            quantized=quantized,
            hard=hard_spatial,
            indices=indices.view(shape[0], shape[2], shape[3]),
            likelihoods=likelihoods,
            log_likelihoods=log_likelihoods,
            aux={"perplexity": perplexity},
        )

    def hard_quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_flat, shape = _flatten_z(z)
        indices = self._nearest_code(z_flat)
        hard = F.embedding(indices, self.codebook)
        return _unflatten_z(hard, shape), indices.view(shape[0], shape[2], shape[3])

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() != 3:
            raise ValueError("Expected indices with shape [N, H, W]")
        flat = indices.reshape(-1)
        vectors = F.embedding(flat, self.codebook)
        n, h, w = indices.shape
        return vectors.view(n, h, w, self.dim).permute(0, 3, 1, 2).contiguous()

    def build_cdf(self, precision: int = 16) -> Tuple[List[int], List[int], List[int]]:
        probs = self.ema_probs.clone()
        probs = probs / probs.sum().clamp_min(1e-9)
        total = 1 << precision
        pmf = torch.clamp((probs * total).round().long(), min=1)
        diff = total - pmf.sum()
        pmf[0] += diff
        cdf = torch.zeros(self.num_codes + 1, dtype=torch.int32)
        cdf[1:] = torch.cumsum(pmf, dim=0)
        return [cdf.tolist()], [len(cdf)], [0]

    def _warmup_passthrough(
        self, z: torch.Tensor, shape: Tuple[int, int, int, int]
    ) -> QuantizationResult:
        ones = torch.ones(shape[0], 1, shape[2], shape[3], device=z.device, dtype=z.dtype)
        zeros = torch.zeros(shape[0], shape[2], shape[3], device=z.device, dtype=torch.long)
        aux = {"perplexity": torch.tensor(float(self.num_codes), device=z.device)}
        log_ones = torch.zeros_like(ones)
        return QuantizationResult(z, z, zeros, ones, log_ones, aux)

    @torch.no_grad()
    def _cache_latents(self, z_flat: torch.Tensor) -> None:
        if z_flat.numel() == 0:
            return
        target = self.num_codes * self.init_samples_per_code
        if target <= 0:
            return
        if z_flat.size(0) > target:
            perm = torch.randperm(z_flat.size(0), device=z_flat.device)[:target]
            snapshots = z_flat[perm]
        else:
            snapshots = z_flat
        snapshots = snapshots.detach().cpu()
        self._init_cache.append(snapshots)
        self._cached_latents = min(self._cached_latents + snapshots.shape[0], target * len(self._init_cache))

    @torch.no_grad()
    def _initialize_codebook(self) -> None:
        if self._initialized:
            return
        if self._init_cache:
            latents = torch.cat(list(self._init_cache), dim=0)
            if latents.numel() == 0:
                self.codebook.data.normal_()
            else:
                samples_per_code = self.init_samples_per_code
                needed = self.num_codes * samples_per_code
                if latents.size(0) < needed:
                    repeat = needed - latents.size(0)
                    extra = latents[torch.randint(latents.size(0), (repeat,))]
                    latents = torch.cat([latents, extra], dim=0)
                perm = torch.randperm(latents.size(0))
                latents = latents[perm][: needed]
                init_vectors = (
                    latents.view(self.num_codes, samples_per_code, self.dim)
                    .mean(dim=1)
                    .to(self.codebook.device)
                )
                self.codebook.data.copy_(init_vectors)
        else:
            self.codebook.data.normal_()
        self._initialized = True
        self._init_cache.clear()
        self._cached_latents = 0

    def _perplexity(self) -> torch.Tensor:
        probs = self.ema_probs.clamp(min=1e-9)
        entropy = -(probs * torch.log(probs)).sum()
        return torch.exp(entropy)

    def _lookup_probs(self, indices: torch.Tensor) -> torch.Tensor:
        probs = torch.gather(self.ema_probs, 0, indices)
        return probs.clamp(min=1e-9)

    def _reshape_probs(self, probs: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        n, _, h, w = shape
        return probs.view(n, h, w, 1).permute(0, 3, 1, 2).contiguous()

    def _nearest_code(self, z_flat: torch.Tensor) -> torch.Tensor:
        z_norm = (z_flat ** 2).sum(dim=1, keepdim=True)
        code_norm = (self.codebook ** 2).sum(dim=1)
        distances = z_norm + code_norm - 2 * z_flat @ self.codebook.t()
        return torch.argmin(distances, dim=1)

    def _update_usage(self, indices: torch.Tensor) -> None:
        counts = torch.bincount(indices, minlength=self.num_codes).float()
        total = counts.sum().clamp_min(1.0)
        probs = counts / total
        self.ema_probs.mul_(self.prob_decay).add_(probs * (1.0 - self.prob_decay))
        self.usage_acc.add_(counts)
        self._counts_since_replace.add_(counts)

    def _current_replacement_freq(self, iteration: Optional[int]) -> Optional[int]:
        if iteration is None:
            return None
        for start, end, freq in self.replace_schedule:
            if start <= iteration <= end:
                return freq
        return None

    def _maybe_replace_codewords(self, iteration: Optional[int]) -> None:
        freq = self._current_replacement_freq(iteration)
        if freq is None or iteration is None:
            return
        if iteration - self._last_replacement_iter < freq:
            return
        counts = self._counts_since_replace.clone()
        total = counts.sum().item()
        if total == 0:
            return
        probs = counts / counts.sum().clamp_min(1e-9)
        inactive = probs < self.replace_threshold
        active = probs >= self.replace_threshold
        if inactive.sum() == 0 or active.sum() == 0:
            self._counts_since_replace.zero_()
            self._last_replacement_iter = iteration
            return
        weights = probs[active]
        weights = weights / weights.sum().clamp_min(1e-9)
        active_idx = active.nonzero(as_tuple=False).squeeze(1)
        inactive_idx = inactive.nonzero(as_tuple=False).squeeze(1)
        for target in inactive_idx:
            sampled_pos = torch.multinomial(weights, 1).item()
            src = active_idx[sampled_pos]
            noise = 0.01 * torch.randn_like(self.codebook.data[src])
            self.codebook.data[target].copy_(self.codebook.data[src] + noise)
        self._counts_since_replace.zero_()
        self._last_replacement_iter = iteration

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        self._initialized = True

    def _differentiable_projection(
        self, z_flat: torch.Tensor, indices: torch.Tensor, hard: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class DiVeQQuantizer(DifferentiableVectorQuantizer):
    """Differentiable VQ following Eq. (8) in the paper."""

    def _differentiable_projection(
        self, z_flat: torch.Tensor, indices: torch.Tensor, hard: torch.Tensor
    ) -> torch.Tensor:
        direction = hard - z_flat
        if self.detach_variant:
            norm = direction.norm(dim=1, keepdim=True).clamp(min=1e-8)
            unit = (direction / norm).detach()
            return z_flat + norm * unit

        if self.training:
            noise = torch.randn_like(direction) * self.sigma
        else:
            noise = torch.zeros_like(direction)
        v_d = direction + noise
        dir_norm = v_d.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scale = direction.norm(dim=1, keepdim=True)
        unit = (v_d / dir_norm).detach()
        return z_flat + scale * unit


class SFDiveQQuantizer(DifferentiableVectorQuantizer):
    """Space-Filling DiVeQ as in Eq. (12)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, use_replacement=False, **kwargs)

    def _differentiable_projection(
        self, z_flat: torch.Tensor, indices: torch.Tensor, hard: torch.Tensor
    ) -> torch.Tensor:
        if self.num_codes < 2:
            return hard
        device = z_flat.device
        dtype = z_flat.dtype
        num_pairs = self.num_codes - 1
        lambdas = torch.rand(num_pairs, device=device, dtype=dtype)
        c0 = self.codebook[:-1]
        c1 = self.codebook[1:]
        dithered = (1 - lambdas.unsqueeze(1)) * c0 + lambdas.unsqueeze(1) * c1
        z_norm = (z_flat ** 2).sum(dim=1, keepdim=True)
        dither_norm = (dithered ** 2).sum(dim=1)
        distances = z_norm + dither_norm - 2 * z_flat @ dithered.t()
        pair_indices = torch.argmin(distances, dim=1)
        lambda_selected = lambdas[pair_indices].unsqueeze(1)
        c_i = c0[pair_indices]
        c_ip1 = c1[pair_indices]

        dir_i = c_i - z_flat
        dir_ip1 = c_ip1 - z_flat
        norm_i = dir_i.norm(dim=1, keepdim=True)
        norm_ip1 = dir_ip1.norm(dim=1, keepdim=True)

        if self.detach_variant:
            unit_i = (((1 - lambda_selected) * dir_i) / norm_i.clamp(min=1e-8)).detach()
            unit_ip1 = ((lambda_selected * dir_ip1) / norm_ip1.clamp(min=1e-8)).detach()
            return z_flat + norm_i * unit_i + norm_ip1 * unit_ip1

        if self.training:
            noise = torch.randn_like(z_flat) * self.sigma
        else:
            noise = torch.zeros_like(z_flat)

        vd_i = dir_i + noise
        vd_ip1 = dir_ip1 + noise
        unit_i = (((1 - lambda_selected) * vd_i) / vd_i.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        unit_ip1 = ((lambda_selected * vd_ip1) / vd_ip1.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()

        return z_flat + norm_i * unit_i + norm_ip1 * unit_ip1
