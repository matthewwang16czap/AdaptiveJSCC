import torchvision
import torch
import torch.distributed as dist
import json
from utils.data_utils import get_batch_data
from utils.logger_utils import get_logger_dir
from utils.universal_utils import get_path


def test(net, test_loader, logger, config):
    net.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    metric_names = [
        "psnr",
        "ssim",
        "msssim",
        "snr",
        "cbr",
    ]
    results = []
    for snr in config.snrs:
        for cbr in config.cbrs:
            metrics = {name: 0.0 for name in metric_names}
            counts = 0
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):
                    input, valid, hr_input = get_batch_data(data, config)
                    if snr == config.snrs[0] and cbr == config.cbrs[0]:
                        save_path = get_path(
                            ".", "recons", f"origin_{rank}_{batch_idx}.png"
                        )
                        if config.sr:
                            torchvision.utils.save_image(hr_input[0], save_path)
                        else:
                            torchvision.utils.save_image(input[0], save_path)
                    (
                        recon_images,
                        [_, _],
                        [mse, psnr, ssim, msssim],
                        img_loss,
                    ) = net(input, valid, hr_input, snr, cbr)
                    # for visualization
                    if rank == 0:
                        save_path = get_path(
                            ".",
                            "recons",
                            f"recon_{rank}_{batch_idx}_{snr}_{cbr}.png",
                        )
                        torchvision.utils.save_image(recon_images[0], save_path)
                    # Update batch data to metrics
                    batch_size = input.size(0)
                    metrics["psnr"] += psnr.item() * batch_size
                    metrics["ssim"] += ssim.item() * batch_size
                    metrics["msssim"] += msssim.item() * batch_size
                    counts += batch_size
            # DDP reduce
            for key in metrics:
                tensor = torch.tensor(metrics[key], device=config.device)
                if dist.is_initialized():
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                metrics[key] = (tensor / (counts * world_size)).item()
            # Store result
            results.append(
                {
                    "snr": snr,
                    "cbr": cbr,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "msssim": metrics["msssim"],
                }
            )
    # logging
    if rank == 0 and logger is not None:
        logger.info("Start Test:")
        logger.info(
            f"{'SNR':>8}"
            f"{'CBR':>12}"
            f"{'PSNR':>10}"
            f"{'SSIM':>10}"
            f"{'MS-SSIM':>10}"
        )
        for r in results:
            logger.info(
                f"{r['snr']:>8.2f}"
                f"{r['cbr']:>12.4f}"
                f"{r['psnr']:>10.3f}"
                f"{r['ssim']:>10.3f}"
                f"{r['msssim']:>10.3f}"
            )
        logger.info("Finish Test!")
        with open(get_path(get_logger_dir(logger), "test_results.json"), "w") as f:
            json.dump(results, f, indent=4)
    return results
