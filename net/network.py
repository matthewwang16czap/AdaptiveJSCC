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
from .modules import Attractor
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
        self.attractors = (
            Attractor(
                channels=encoder_kwargs["embed_dims"][-1],
                in_depth=3,
                out_depth=3,
            )
            if config.attractor
            else None
        )
        self.sr = SRNet(ckpt_path="./pretrained/dmnet_x2.pth") if config.sr else None

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(
        self, input_image, valid, hr_input_image, given_SNR=None, given_rate=None
    ):
        if given_SNR is None:
            SNR = choice(self.config.multiple_snr)
            SNR = torch.tensor([SNR], device=input_image.device)
            # SNR = sample_choice_ddp(self.multiple_snr, input_image.device)
        else:
            SNR = torch.tensor([given_SNR], device=input_image.device)
        chan_param = SNR
        if given_rate is None:
            channel_number = choice(self.config.channel_number)
            channel_number = torch.tensor([channel_number], device=input_image.device)
            # channel_number = sample_choice_ddp(self.channel_number, input_image.device)
        else:
            channel_number = torch.tensor([given_rate], device=input_image.device)
        feature, mask, feature_H, feature_W = self.encoder(
            input_image, SNR, channel_number, self.config.model
        )
        if not self.config.training:
            # has proven 8 bit quantization doesn't affect result a lot
            feature, scale = quantize_symmetric(feature, bits=self.config.quant_bits)
        CBR = (
            channel_number
            * feature.shape[1]
            * (self.config.quant_bits / 8)
            / input_image[0].numel()
        )
        with torch.autocast(device_type=input_image.device.type, enabled=False):
            avg_pwr = (
                (feature.float().pow(2).sum()) / mask.float().sum().clamp(min=1.0)
            ).clamp(min=-1e9)
        if self.config.pass_channel:
            noisy_feature = self.feature_pass_channel(feature.float(), SNR, avg_pwr)
        else:
            noisy_feature = feature
        if not self.training:
            noisy_feature = dequantize_symmetric(noisy_feature, scale)
        noisy_feature = noisy_feature * mask
        # Pass noisy feature through feature_denoiser network
        if self.attractors is not None:
            restored_feature = self.attractors(noisy_feature, mask, SNR)
            pred_noise = noisy_feature - restored_feature
            # repredict chan_param
            with torch.autocast(device_type=input_image.device.type, enabled=False):
                signal_power = (feature.float() * mask.float()).pow(
                    2
                ).sum() / mask.float().sum().clamp(min=1)
                restore_mse = self.feature_mse_loss(
                    restored_feature.float(),
                    feature.float(),
                    mask.float(),
                )
                ratio = signal_power / restore_mse
                chan_param = 10.0 * torch.log10(ratio).detach()
        else:
            pred_noise = torch.zeros_like(noisy_feature)
            restored_feature = noisy_feature
        recon_image = self.decoder(
            restored_feature, chan_param, self.config.model, feature_H, feature_W, valid
        )
        # Super Resolution
        if self.sr is not None:
            recon_image = self.sr(recon_image.detach())
        # Compute loss and metrics
        if self.sr is not None:
            img_loss, mse, psnr = self.mse_loss(recon_image, hr_input_image, valid)
        else:
            img_loss, mse, psnr = self.mse_loss(recon_image, input_image, valid)
        img_loss = img_loss.mean()
        # rescale to [0,255] loss to avoid too small loss
        img_loss = img_loss * 255 * 255
        mse = mse.mean()
        psnr = psnr.mean()
        if self.sr is not None:
            ssim = self.ssim(recon_image, hr_input_image).mean().detach()
            msssim = self.msssim(recon_image, hr_input_image).mean().detach()
        else:
            ssim = self.ssim(recon_image, input_image).mean().detach()
            msssim = self.msssim(recon_image, input_image).mean().detach()
        return (
            recon_image,
            [restored_feature, pred_noise, noisy_feature, feature],
            [mask, feature_H, feature_W],
            [
                CBR.item() if isinstance(CBR, torch.Tensor) else CBR,
                SNR.item() if isinstance(SNR, torch.Tensor) else SNR,
                (
                    chan_param.item()
                    if isinstance(chan_param, torch.Tensor)
                    else chan_param
                ),
            ],
            [mse, psnr, ssim, msssim],
            img_loss,
        )
