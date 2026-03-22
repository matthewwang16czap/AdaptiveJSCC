from typing import Optional
import torch


@torch.jit.script
def masked_mse(
    X: torch.Tensor,
    Y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    normalization: bool = False,
):
    """
    Returns per-image MSE. Shape: (B,)
    """
    if normalization:
        X = (X + 1) / 2
        Y = (Y + 1) / 2

    diff = (X - Y) ** 2 * (data_range**2)

    if mask is None:
        # mean over C,H,W only
        return diff.mean(dim=(1, 2, 3))

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)

    num = (diff * mask).sum(dim=(1, 2, 3))
    den = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return num / den


class MSEWithPSNR(torch.nn.Module):
    def __init__(self, normalization=False, data_range=1.0):
        super().__init__()
        self.normalization = normalization
        self.data_range = data_range

    def forward(self, X, Y, mask=None):
        mse = masked_mse(
            X,
            Y,
            mask,
            data_range=self.data_range,
            normalization=self.normalization,
        )

        psnr_val = 10.0 * torch.log10((self.data_range**2) / mse)

        return mse, mse.detach(), psnr_val.detach()
