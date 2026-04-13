from typing import Optional
import torch
import lpips


@torch.jit.script
def masked_mse(
    X: torch.Tensor,
    Y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    normalized: bool = False,
) -> torch.Tensor:
    """
    Returns per-image MSE. Shape: (B,)

    Args:
        X, Y:       Input tensors (B, C, H, W).
        mask:       Optional float mask (B, H, W) or (B, 1, H, W) in [0, 1].
        data_range: The value range of the output (e.g. 1.0 or 255.0).
        normalized: If True, inputs are already in [0, data_range].
                    If False, inputs are in [-1, 1] and are remapped.
    """
    if not normalized:
        X = (X + 1.0) / 2.0 * data_range
        Y = (Y + 1.0) / 2.0 * data_range
    diff = (X - Y) ** 2
    if mask is None:
        return diff.mean(dim=(1, 2, 3))
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
    num = (diff * mask).sum(dim=(1, 2, 3))
    den = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return num / den


class MSEWithPSNR(torch.nn.Module):
    """
    Returns:
        loss:      Scalar MSE for backprop (mean over batch).
        mse:       Per-image MSE, detached. Shape: (B,)
        psnr:      Per-image PSNR in dB, detached. Shape: (B,)
    """

    def __init__(self, normalized: bool = False, data_range: float = 1.0):
        super().__init__()
        self.normalized = normalized
        self.data_range = data_range

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        mse = masked_mse(
            X,
            Y,
            mask,
            data_range=self.data_range,
            normalized=self.normalized,
        )
        psnr = 10.0 * torch.log10((self.data_range**2) / mse.clamp_min(1e-8))
        # rescale to [0,255] loss to avoid too small loss
        # mse_loss = mse.mean() * 255 * 255
        return mse.mean(), mse.detach(), psnr.detach()


@torch.jit.ignore
def masked_lpips(
    X: torch.Tensor,
    Y: torch.Tensor,
    model: torch.nn.Module,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    normalized: bool = False,
) -> torch.Tensor:
    """
    Returns per-image LPIPS. Shape: (B,)

    Masking strategy: LPIPS is run on the full image (no pixel zeroing,
    which would bias the perceptual score by introducing artificial gray
    regions). The resulting per-image score is then weighted by the mean
    mask value so fully-masked images contribute ~0 loss.

    Args:
        X, Y:       Input tensors (B, C, H, W).
        model:      LPIPS model instance.
        mask:       Optional float mask (B, H, W) or (B, 1, H, W) in [0, 1].
        data_range: The value range of the inputs (e.g. 1.0 or 255.0).
        normalized: If True, inputs are in [0, data_range] and are remapped
                    to [-1, 1] for LPIPS. If False, inputs are already [-1, 1].
    """
    if normalized:
        X = (X / data_range) * 2.0 - 1.0
        Y = (Y / data_range) * 2.0 - 1.0
    # normalized=False: inputs are already in [-1, 1], nothing to do.
    per_image = model(X, Y).view(X.size(0))  # (B,)
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        # Weight each image's score by its mask coverage instead of
        # zeroing pixels, which would corrupt the perceptual features.
        mask_weight = mask.mean(dim=(1, 2, 3))  # (B,)
        per_image = per_image * mask_weight
    return per_image


class LPIPSWithScore(torch.nn.Module):
    """
    Returns:
        loss:       Scalar LPIPS for backprop (mean over batch).
        lpips_val:  Per-image LPIPS, detached. Shape: (B,)
    """

    def __init__(self, normalized: bool = False, data_range: float = 1.0):
        super().__init__()
        self.normalized = normalized
        self.data_range = data_range
        self.lpips_model = lpips.LPIPS(net="alex")

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        lpips_val = masked_lpips(
            X,
            Y,
            self.lpips_model,
            mask,
            data_range=self.data_range,
            normalized=self.normalized,
        )
        return lpips_val.mean(), lpips_val.detach()


class ImageLoss(torch.nn.Module):
    """
    Combined image loss: lambda_mse * MSE + lambda_lpips * LPIPS.

    Returns:
        loss:       Scalar combined loss for backprop.
        metrics:    Dict of detached per-image tensors for logging:
                      - "mse"   shape (B,)
                      - "psnr"  shape (B,)
                      - "lpips" shape (B,)
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_lpips: float = 1.0,
        normalized: bool = False,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_lpips = lambda_lpips
        self.mse_loss = (
            MSEWithPSNR(normalized=normalized, data_range=data_range)
            if lambda_mse > 0
            else None
        )
        self.lpips_loss = (
            LPIPSWithScore(normalized=normalized, data_range=data_range)
            if lambda_lpips > 0
            else None
        )

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        mse_loss, mse, psnr = (
            self.mse_loss(X, Y, mask) if self.mse_loss is not None else (0, 0, 0)
        )
        lpips_loss, lpips_val = (
            self.lpips_loss(X, Y, mask) if self.lpips_loss is not None else (0, 0)
        )
        loss = self.lambda_mse * mse_loss + self.lambda_lpips * lpips_loss
        metrics = {"mse": mse.mean(), "psnr": psnr.mean(), "lpips": lpips_val.mean()}
        return loss, metrics
