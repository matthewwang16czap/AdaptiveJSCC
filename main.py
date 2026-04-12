import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from net.network import SwinJSCC
from data.datasets import get_loader
from configs.config import Config
from tools.test import test
from tools.train import train_one_epoch
from utils.ddp_utils import cleanup_ddp, initialize_ddp
from utils.logger_utils import logger_configuration
from utils.parser_utils import create_parser
from utils.torch_utils import load_weights, save_model, seed_torch
from utils.model_utils import freeze_model

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


if __name__ == "__main__":
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_capability())
    # Parse args
    parser = create_parser()
    args = parser.parse_args()
    # Initialize distributed
    ddp_env = initialize_ddp()
    # Config and setup
    config = Config(args)
    config.device = ddp_env["device"]
    config.device_id = ddp_env["device_id"]
    # config.batch_size = config.batch_size // ddp_env["world_size"]
    config.accum_steps = 1
    base_seed = 42 + ddp_env["rank"]  # Different seed per GPU
    seed_torch(base_seed)
    logger = logger_configuration(config, save_log=(ddp_env["rank"] == 0))
    if ddp_env["rank"] == 0:
        logger.info(config.__dict__)
    # Model
    net = SwinJSCC(config).to(device=config.device)
    # Print model if needs
    # if ddp_env["rank"] == 0:
    #     logger.info(net)
    # Load weights if needed
    if config.pretrained_model_path is not None:
        if ddp_env["rank"] == 0:
            load_weights(net, config.pretrained_model_path)
        # DDP initialization will sync these weights automatically
    # Freeze params
    freeze_model(net, config)
    trainable = [name for name, p in net.named_parameters() if p.requires_grad]
    logger.info("\n".join(trainable))
    # DDP wrap model
    if ddp_env["world_size"] > 1:
        net = DDP(
            net,
            device_ids=[config.device_id],
            output_device=config.device_id,
            find_unused_parameters=False,
        )
    # Check if model is wrapped in DDP
    is_ddp = hasattr(net, "module")
    model = net.module if is_ddp else net
    # Data
    train_loader, test_loader, train_sampler, test_sampler = get_loader(
        config, rank=ddp_env["rank"], world_size=ddp_env["world_size"]
    )
    # Optimizer Effective lr
    # lr = (
    #     ddp_env["world_size"] * config.learning_rate
    #     if ddp_env["world_size"] > 1
    #     else config.learning_rate
    # )
    lr = config.learning_rate
    model_params = [{"params": model.parameters(), "lr": lr}]
    optimizer = optim.Adam(model_params, lr=lr)
    scaler = torch.amp.GradScaler() if config.amp else None
    global_step = 0
    steps_epoch = global_step // len(train_loader)
    # Train or Test
    if config.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if test_sampler is not None:
                test_sampler.set_epoch(epoch)
            global_step = train_one_epoch(
                epoch,
                global_step,
                net,
                train_loader,
                optimizer,
                logger,
                config,
                scaler,
            )
            # Save/check only on rank 0
            if (epoch + 1) % config.save_model_freq == 0:
                if ddp_env["rank"] == 0:
                    save_model(
                        net.module if isinstance(net, DDP) else net,
                        save_path=f"{config.models_dir}/{config.save_filename}_EP{epoch + 1}.model",
                    )
                test(
                    net,
                    test_loader,
                    (
                        logger
                        if (not isinstance(net, DDP)) or (ddp_env["rank"] == 0)
                        else None
                    ),
                    config,
                )
                if dist.is_initialized():
                    dist.barrier(device_ids=[config.device_id])
    else:
        test(
            net,
            test_loader,
            (logger if (not isinstance(net, DDP)) or (ddp_env["rank"] == 0) else None),
            config,
        )
    # DDP CLEANUP
    cleanup_ddp()
