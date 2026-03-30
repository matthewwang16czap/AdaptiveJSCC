import torch
import torch.nn as nn


class Channel(nn.Module):
    """
    Physically complex baseband channel.
    Input shape: (B, N, C), where C = 2 * num_complex_dims
    """

    def __init__(self, config):
        super().__init__()
        self.chan_type = config.channel_type
        self.device = config.device
        if config.logger:
            config.logger.info(
                f"【Channel】: Built {self.chan_type} channel, SNR {config.snrs} dB."
            )

    # Utility: packed (real, imag) → complex
    @staticmethod
    def _to_complex(x):
        B, N, C = x.shape
        assert C % 2 == 0, "Channel input C must be even (real + imag)"
        C2 = C // 2
        real = x[..., :C2]
        imag = x[..., C2:]
        return torch.complex(real, imag)

    # Utility: complex → packed (real, imag)
    @staticmethod
    def _to_packed(xc):
        return torch.cat([xc.real, xc.imag], dim=-1)

    # Power normalization (complex baseband)
    @staticmethod
    def _normalize_power(xc, target_power=1.0, eps=1e-12):
        power = torch.mean(torch.abs(xc) ** 2)
        scale = torch.sqrt(torch.tensor(target_power, device=xc.device) / (power + eps))
        return xc * scale, power

    # AWGN channel
    @staticmethod
    def _awgn(xc, snr_db):
        snr = torch.pow(torch.tensor(10.0, device=xc.device), snr_db / 10)
        sigma = torch.sqrt(1.0 / (2.0 * snr))
        noise = torch.complex(
            torch.randn_like(xc.real) * sigma,
            torch.randn_like(xc.imag) * sigma,
        )
        return xc + noise

    # Rayleigh fading channel
    @staticmethod
    def _rayleigh(xc, snr_db):
        h = torch.complex(
            torch.randn_like(xc.real),
            torch.randn_like(xc.imag),
        ) / torch.sqrt(torch.tensor(2.0, device=xc.device))
        snr = torch.pow(torch.tensor(10.0, device=xc.device), snr_db / 10)
        sigma = torch.sqrt(1.0 / (2.0 * snr))
        noise = torch.complex(
            torch.randn_like(xc.real) * sigma,
            torch.randn_like(xc.imag) * sigma,
        )
        return h * xc + noise

    # Forward
    def forward(self, x, snr_db, avg_pwr=None):
        """
        x: (B, N, C) packed real/imag
        snr_db: scalar tensor or float
        avg_pwr: optional external power normalization scalar
        """
        # Convert to complex correctly
        xc = self._to_complex(x)
        # Normalize power
        if avg_pwr is not None:
            # External power reference
            xc = xc / torch.sqrt(avg_pwr)
        else:
            xc, _ = self._normalize_power(xc, target_power=1.0)
        # Channel
        if self.chan_type == 0 or self.chan_type == "none":
            yc = xc
        elif self.chan_type == 1 or self.chan_type == "awgn":
            yc = self._awgn(xc, snr_db)
        elif self.chan_type == 2 or self.chan_type == "rayleigh":
            yc = self._rayleigh(xc, snr_db)
        else:
            raise ValueError(f"Unsupported channel type: {self.chan_type}")
        # Physically correct output
        y = self._to_packed(yc)
        # Restore original power scale if needed
        if avg_pwr is not None:
            y = y * torch.sqrt(avg_pwr)
        return y
