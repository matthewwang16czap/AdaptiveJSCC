from utils.universal_utils import AverageMeter
from utils.data_utils import get_batch_data
import torch
import time


def train_one_step(
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
    # train batch data
    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        input, valid = get_batch_data(data, config)
        # Forward and backward pass
        with torch.autocast(
            device_type=input.device.type, enabled=(scaler is not None)
        ):
            (
                recon_images,
                [cbr, snr],
                metrics,
                img_loss,
            ) = net(input, valid)
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
        metrics = {
            "elapsed": time.time() - start_time,
            "loss": img_loss.item(),
            "snr": snr.item(),
            "cbr": cbr.item(),
            **metrics,
        }
        # Logging
        if global_step % config.print_step == 0:
            process = (
                (global_step % train_loader.__len__())
                / (train_loader.__len__())
                * 100.0
            )
            line1 = [
                f"Epoch {epoch + 1}",
                f"Step [{batch_idx + 1}/{len(train_loader)}={process:.2f}%]",
                f"Time {metrics['elapsed']:.3f}",
                f"Lr {config.learning_rate}",
            ]
            line2 = [
                f"Loss {metrics['loss']:>10.2e}",
                f"SNR  {metrics['snr']:>10.2f}",
                f"CBR  {metrics['cbr']:>10.2f}",
                f"PSNR {metrics['psnr']:>10.2f}",
                f"MSE  {metrics['mse']:>10.2e}",
                f"LPIPS {metrics['lpips']:>10.2e}",
                f"SSIM {metrics['ssim']:>10.2e}",
                f"MSSSIM {metrics['msssim']:>10.2e}",
            ]
            logger.info(" | ".join(line1))
            logger.info(" | ".join(line2))
    return global_step
