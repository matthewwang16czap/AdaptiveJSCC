from .decoder import create_decoder
from .encoder import create_encoder
from loss.image_losses import ImageLoss
from utils.model_utils import quantize_symmetric, dequantize_symmetric
from .channel import Channel
from random import choice
import torch
import torch.nn as nn
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)


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
        self.channel = Channel(config)
        self.mse_loss = ImageLoss(
            config.lambda_mse, config.lambda_lpips, normalized=True, data_range=1.0
        )
        self.ssim = SSIM(data_range=1.0)
        self.msssim = MS_SSIM(data_range=1.0)
        # feature_channels = encoder_kwargs["embed_dims"][-1]

    def feature_pass_channel(self, feature, snr_db, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, snr_db, avg_pwr)
        return noisy_feature

    def forward(self, input_image, valid, given_snr=None, given_cbr=None):
        if given_snr is None:
            snr = choice(self.config.snrs)
            snr = torch.tensor([snr], device=input_image.device)
        else:
            snr = torch.tensor([given_snr], device=input_image.device)
        if given_cbr is None:
            cbr = choice(self.config.cbrs)
            cbr = torch.tensor([cbr], device=input_image.device)
        else:
            cbr = torch.tensor([given_cbr], device=input_image.device)
        feature, feature_H, feature_W = self.encoder(
            input_image, snr, cbr, self.config.token_channel_balance_ratio
        )
        mask = feature != 0
        if not self.training:
            # has proven 8 bit quantization doesn't affect result a lot
            feature, scale = quantize_symmetric(feature, bits=self.config.quant_bits)
        with torch.autocast(device_type=input_image.device.type, enabled=False):
            avg_pwr = (
                (
                    (feature.float().pow(2).sum()) / mask.float().sum().clamp(min=1.0)
                ).clamp(min=-1e9)
                if mask.sum() < feature.numel()
                else (feature.float().pow(2).mean())
            )
        if self.config.pass_channel:
            noisy_feature = self.feature_pass_channel(feature.float(), snr, avg_pwr)
        else:
            noisy_feature = feature
        if not self.training:
            noisy_feature = dequantize_symmetric(noisy_feature, scale)
        noisy_feature = noisy_feature * mask
        recon_images = self.decoder(noisy_feature, snr, feature_H, feature_W, valid)
        # losses
        img_loss, metrics = self.mse_loss(recon_images, input_image, valid)
        # metrics
        mse = metrics["mse"].mean()
        psnr = metrics["psnr"].mean()
        lpips = metrics["lpips"].mean()
        ssim = self.ssim(recon_images, input_image).mean().detach()
        msssim = self.msssim(recon_images, input_image).mean().detach()
        return (
            recon_images,
            [cbr, snr],
            [mse, psnr, lpips, ssim, msssim],
            img_loss,
        )
