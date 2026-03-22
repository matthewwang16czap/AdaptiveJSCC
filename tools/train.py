from utils.universal_utils import AverageMeter
from utils.data_utils import get_batch_data
import torch
import time


def train_one_epoch(
    epoch,
    global_step,
    net,
    train_loader,
    optimizer,
    logger,
    config,
    scaler,
):
    is_ddp = hasattr(net, "module")
    net.train()
    optimizer.zero_grad(set_to_none=True)
    # Initialize metrics
    metrics_names = ["elapsed", "losses", "psnrs", "ssims", "msssims", "cbrs", "snrs"]
    metrics = {name: AverageMeter() for name in metrics_names}
    # train batch data
    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        input, valid, hr_input = get_batch_data(data, config)
        # Forward and backward pass
        with torch.autocast(
            device_type=input.device.type, enabled=(scaler is not None)
        ):
            (
                recon_image,
                [restored_feature, pred_noise, noisy_feature, feature],
                [mask, feature_H, feature_W],
                [CBR, SNR, chan_param],
                [mse, psnr, ssim, msssim],
                img_loss,
            ) = net(input, valid, hr_input)
            img_loss = img_loss / config.accum_steps
        if scaler is not None:
            if is_ddp and (batch_idx % config.accum_steps != config.accum_steps - 1):
                # During accumulation, avoid syncing DDP gradients
                with net.no_sync():
                    scaler.scale(img_loss).backward()
            else:
                scaler.scale(img_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            if is_ddp and (batch_idx % config.accum_steps != config.accum_steps - 1):
                with net.no_sync():
                    img_loss.backward()
            else:
                img_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        # Update metrics
        metrics["elapsed"].update(time.time() - start_time)
        metrics["losses"].update(img_loss.item())
        metrics["cbrs"].update(CBR)
        metrics["snrs"].update(SNR)
        metrics["psnrs"].update(psnr.item())
        metrics["ssims"].update(ssim.item())
        metrics["msssims"].update(msssim.item())
        # Logging
        if global_step % config.print_step == 0:
            process = (
                (global_step % train_loader.__len__())
                / (train_loader.__len__())
                * 100.0
            )
            log_components = [
                f"Epoch {epoch + 1}",
                f"Step [{batch_idx + 1}/{len(train_loader)}={process:.2f}%]",
                f"Time {metrics['elapsed'].val:.3f}",
                f"Loss {metrics['losses'].val:.2e}",
                f"CBR {metrics['cbrs'].val:.4f}",
                f"SNR {metrics['snrs'].val:.1f}",
                f"PSNR {metrics['psnrs'].val:.3f}",
                f"SSIM {metrics['ssims'].val:.3f}",
                f"MSSSIM {metrics['msssims'].val:.3f}",
                f"Lr {config.learning_rate}",
            ]
            logger.info(" | ".join(log_components))
            # Reset metrics after logging
            for metric in metrics.values():
                metric.clear()
    return global_step


def train_attractors_one_epoch(
    epoch,
    global_step,
    net,
    train_loader,
    optimizer,
    logger,
    config,
    scaler,
):
    net.train()
    # Check if model is wrapped in DDP
    is_ddp = hasattr(net, "module")
    model = net.module if is_ddp else net
    # Initialize metric meters
    metric_names = [
        "elapsed",
        "losses",
        "cbrs",
        "snrs",
        "chan_params",
        "psnrs",
        "ssims",
        "msssims",
    ]
    metrics = {name: AverageMeter() for name in metric_names}
    # Train batch data
    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        input, valid, hr_input = get_batch_data(data, config)
        # Forward pass
        optimizer.zero_grad()
        with torch.autocast(
            device_type=input.device.type, enabled=(scaler is not None)
        ):
            (
                recon_image,
                [restored_feature, pred_noise, noisy_feature, feature],
                [mask, feature_H, feature_W],
                [CBR, SNR, chan_param],
                [mse, psnr, ssim, msssim],
                img_loss,
            ) = net(input, valid, hr_input)
            feature = feature.detach()
            noisy_feature = noisy_feature.detach()
            mask = mask.detach()
            noise = noisy_feature - feature
            # Loss Components
            # (1) Orthogonal loss: encourage pred_noise ⟂ restored_feature
            orth_loss = model.feature_orthogonal_loss(
                restored_feature, noise, pred_noise, mask
            )
            # (2) MSE between restored_feature and ground-truth feature
            mse_loss = model.feature_mse_loss(restored_feature, feature, mask, noise)
            # (3) Self-consistency: D(feature + pred_noise) ≈ feature
            restored_twice = model.attractors(
                (feature + pred_noise).detach(), mask, SNR
            )
            self_loss = model.feature_mse_loss(
                restored_twice, feature, mask, pred_noise
            )
            a_1, a_2, a_3, a_4 = config.alpha_losses
            total_loss = (
                a_1 * orth_loss + a_2 * mse_loss + a_3 * self_loss + a_4 * img_loss
            )
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        # Metric computation
        metrics["elapsed"].update(time.time() - start_time)
        metrics["losses"].update(total_loss.item())
        metrics["cbrs"].update(CBR)
        metrics["snrs"].update(SNR)
        metrics["psnrs"].update(psnr.item())
        metrics["ssims"].update(ssim.item())
        metrics["msssims"].update(msssim.item())
        metrics["chan_params"].update(chan_param)
        # Logging
        if global_step % config.print_step == 0:
            process = (
                (global_step % train_loader.__len__())
                / (train_loader.__len__())
                * 100.0
            )
            log_components = [
                f"Epoch {epoch + 1}",
                f"Step [{batch_idx + 1}/{len(train_loader)}={process:.2f}%]",
                f"Time {metrics['elapsed'].val:.3f}",
                f"Loss {metrics['losses'].val:.2e}",
                f"CBR {metrics['cbrs'].val:.4f}",
                f"SNR {metrics['snrs'].val:.2f}",
                f"SNR(denoised) {metrics['chan_params'].val:.2f}",
                f"PSNR {metrics['psnrs'].val:.3f}",
                f"SSIM {metrics['ssims'].val:.3f}",
                f"MSSSIM {metrics['msssims'].val:.3f}",
                f"Orth {orth_loss.item():.4f}",
                f"MSE {mse_loss.item():.4f}",
                f"Recon {img_loss.item():.4f}",
            ]
            logger.info(" | ".join(log_components))
            # Reset metrics after each print interval
            for m in metrics.values():
                m.clear()
    return global_step
