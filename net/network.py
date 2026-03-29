from .decoder import create_decoder
from .encoder import create_encoder
from loss.image_losses import *
from loss.feature_losses import *
from .channel import Channel
from random import choice
import torch
import torch.nn as nn
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
from .sr.sr import SRNet
from utils.torch_utils import *


class SwinJSCC(nn.Module):
    def __init__(self, config):
        super(SwinJSCC, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.feature_mse_loss = FeatureMSELoss()
        self.feature_orthogonal_loss = FeatureOrthogonalLoss(alpha=0.8)
        self.channel = Channel(config)
        self.mse_loss = MSEWithPSNR(normalization=False)
        self.ssim = SSIM(data_range=1.0)
        self.msssim = MS_SSIM(data_range=1.0)
        # feature_channels = encoder_kwargs["embed_dims"][-1]
        self.sr = SRNet(ckpt_path="./pretrained/dmnet_x2.pth") if config.sr else None

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(
        self, input_image, valid, hr_input_image, given_snr=None, given_rate=None
    ):
        if given_snr is None:
            snr = choice(self.config.multiple_snr)
            snr = torch.tensor([snr], device=input_image.device)
        else:
            snr = torch.tensor([given_snr], device=input_image.device)
        if given_rate is None:
            rate = choice(self.config.rate)
            rate = torch.tensor([rate], device=input_image.device)
        else:
            rate = torch.tensor([given_rate], device=input_image.device)
        feature, mask, feature_H, feature_W = self.encoder(input_image, snr, rate)
        if not self.config.training:
            # has proven 8 bit quantization doesn't affect result a lot
            feature, scale = quantize_symmetric(feature, bits=self.config.quant_bits)
        cbr = (
            rate * feature.numel() * (self.config.quant_bits / 8) / input_image.numel()
        )
        with torch.autocast(device_type=input_image.device.type, enabled=False):
            avg_pwr = (
                (
                    (feature.float().pow(2).sum()) / mask.float().sum().clamp(min=1.0)
                ).clamp(min=-1e9)
                if mask is not None
                else (feature.float().pow(2).mean())
            )
        if self.config.pass_channel:
            noisy_feature = self.feature_pass_channel(feature.float(), snr, avg_pwr)
        else:
            noisy_feature = feature
        if not self.training:
            noisy_feature = dequantize_symmetric(noisy_feature, scale)
        noisy_feature = noisy_feature * mask if mask is not None else noisy_feature
        recon_images = self.decoder(noisy_feature, snr, feature_H, feature_W, valid)
        # Super Resolution
        if self.sr is not None:
            recon_images = self.sr(recon_images.detach())
        # Compute loss and metrics
        if self.sr is not None:
            img_loss, mse, psnr = self.mse_loss(recon_images, hr_input_image, valid)
        else:
            img_loss, mse, psnr = self.mse_loss(recon_images, input_image, valid)
        img_loss = img_loss.mean()
        # rescale to [0,255] loss to avoid too small loss
        img_loss = img_loss * 255 * 255
        mse = mse.mean()
        psnr = psnr.mean()
        if self.sr is not None:
            ssim = self.ssim(recon_images, hr_input_image).mean().detach()
            msssim = self.msssim(recon_images, hr_input_image).mean().detach()
        else:
            ssim = self.ssim(recon_images, input_image).mean().detach()
            msssim = self.msssim(recon_images, input_image).mean().detach()
        return (
            recon_images,
            [
                cbr.item() if isinstance(cbr, torch.Tensor) else cbr,
                snr.item() if isinstance(snr, torch.Tensor) else snr,
            ],
            [mse, psnr, ssim, msssim],
            img_loss,
        )
