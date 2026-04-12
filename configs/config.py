import os
import torch
import torch.nn as nn
from datetime import datetime


class Config:
    def __init__(self, args):
        # Base setup
        self.seed = 42
        self.pass_channel = (
            args.pass_channel if hasattr(args, "pass_channel") else False
        )
        self.channel_type = args.channel_type  # "awgn" or "rayleigh"
        self.snrs = [float(x) for x in args.snrs.split(",")]
        self.cbrs = [float(x) for x in args.cbrs.split(",")]
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.CUDA else "cpu")
        self.amp = args.amp if hasattr(args, "amp") else False
        self.quant_bits = args.quant_bits if hasattr(args, "quant_bits") else 8

        # Logging and workdir
        self.print_step = 100
        self.plot_step = 10000
        self.save_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.workdir = f"./history/{self.save_filename}"
        self.homedir = "/home/matthewwang16czap/"
        # self.homedir = "/home/ubuntu/"
        self.pretrained_model_path = "./pretrained/base3.model"
        self.log_dir = f"{self.workdir}/Log_{self.save_filename}.log"
        self.samples_dir = f"{self.workdir}/samples"
        self.models_dir = f"{self.workdir}/models"
        self.logger = None

        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # Training details
        self.training = args.training if hasattr(args, "training") else False
        self.learning_rate = 1e-4
        self.tot_epoch = 10_000_000

        # modules enabling
        self.token_pruner = (
            args.token_pruner if hasattr(args, "token_pruner") else False
        )
        self.token_channel_balance_ratio = (
            args.token_channel_balance_ratio
            if hasattr(args, "token_channel_balance_ratio")
            else 0.1
        )
        self.channel_pruner = (
            args.channel_pruner if hasattr(args, "channel_pruner") else False
        )
        self.snr_adapter = args.snr_adapter if hasattr(args, "snr_adapter") else False
        self.training_modules = [x for x in args.training_modules.split(",")]

        # Dataset & architecture setup
        self._setup_dataset(args)

    def _setup_dataset(self, args):
        """Configure dataset and model-specific parameters."""
        self.dataset_type = "RandomResizedCrop"  # or "LetterBox"
        self.max_test_samples = 100
        self.trainset = args.trainset
        self.testset = args.testset
        self.img_size = args.img_size
        # Trainset options
        trainset_map = {
            "DIV2K": [
                self.homedir + "datasets/DIV2K/DIV2K_train_HR",
                self.homedir + "datasets/DIV2K/DIV2K_valid_HR",
            ],
            "COCO": [
                self.homedir + "datasets/coco-2014/train/data",
                self.homedir + "datasets/coco-2017/train/data",
            ],
        }
        self.train_data_dir = trainset_map.get(args.trainset, [])

        # Testset options
        testset_map = {
            "Kodak": [self.homedir + "datasets/Kodak/"],
            "CLIC21": [self.homedir + "datasets/clic2021/test/"],
            "ffhq": [self.homedir + "datasets/ffhq/"],
            "COCO": [
                self.homedir + "datasets/coco-2014/validation/data",
                self.homedir + "datasets/coco-2017/validation/data",
            ],
        }
        self.test_data_dir = testset_map.get(args.testset, [])

        # Model-specific setup
        self.save_model_freq = 10  # epochs
        if self.img_size == 256:
            self._setup_256(args)
        elif self.img_size == 512:
            self._setup_512(args)
        else:
            raise ValueError(f"Unsupported trainset: {args.trainset}")

    def _setup_256(self, args):
        self.image_dims = (3, 256, 256)
        self.batch_size = 4
        self.test_batch_size = 4
        self.downsample = 4  # number of downsampling layers in encoder
        self.patch_size = 2
        self.in_chans = 3
        self.window_size = 8
        self.mlp_ratio = 4.0
        size_map = {
            "small": dict(
                depths=[2, 2, 2, 2],
                num_heads=[4, 6, 8, 10],
                embed_dims=[128, 192, 256, 320],
            ),
            "base": dict(
                depths=[2, 2, 6, 2],
                num_heads=[4, 6, 8, 10],
                embed_dims=[128, 192, 256, 320],
            ),
            "large": dict(
                depths=[2, 2, 18, 2],
                num_heads=[4, 6, 8, 10],
                embed_dims=[128, 192, 256, 320],
            ),
        }
        if args.model_size not in size_map:
            raise ValueError(f"Unknown model_size: {args.model_size}")
        self.encoder_kwargs = dict(
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dims=size_map[args.model_size]["embed_dims"],
            depths=size_map[args.model_size]["depths"],
            num_heads=size_map[args.model_size]["num_heads"],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            quant_bits=self.quant_bits,
            use_snr_adapter=self.snr_adapter,
            use_token_pruner=self.token_pruner,
            use_channel_pruner=self.channel_pruner,
            module_hidden_ratio=1,
        )

        self.decoder_kwargs = dict(
            patch_size=self.patch_size,
            out_chans=self.in_chans,
            embed_dims=size_map[args.model_size]["embed_dims"][::-1],
            depths=size_map[args.model_size]["depths"][::-1],
            num_heads=size_map[args.model_size]["num_heads"][::-1],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_snr_adapter=self.snr_adapter,
            use_token_pruner=self.token_pruner,
            use_channel_pruner=self.channel_pruner,
            module_hidden_ratio=1,
        )

    def _setup_512(self, args):
        self.image_dims = (3, 512, 512)
        self.batch_size = 8
        self.test_batch_size = 8
        self.downsample = 4
        self.patch_size = 4
        self.in_chans = 3
        self.window_size = 8
        self.mlp_ratio = 4.0
        size_map = {
            "base": dict(
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 8, 16],
                embed_dims=[64, 128, 256, 512],
            ),
            "baseline": dict(
                depths=[2, 2, 6, 2],
                num_heads=[4, 6, 8, 10],
                embed_dims=[128, 192, 256, 320],
            ),
        }
        if args.model_size not in size_map:
            raise ValueError(f"Unknown model_size: {args.model_size}")
        self.encoder_kwargs = dict(
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dims=size_map[args.model_size]["embed_dims"],
            depths=size_map[args.model_size]["depths"],
            num_heads=size_map[args.model_size]["num_heads"],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            quant_bits=self.quant_bits,
            use_snr_adapter=self.snr_adapter,
            use_token_pruner=self.token_pruner,
            use_channel_pruner=self.channel_pruner,
            module_hidden_ratio=1,
        )
        self.decoder_kwargs = dict(
            patch_size=self.patch_size,
            out_chans=self.in_chans,
            embed_dims=size_map[args.model_size]["embed_dims"][::-1],
            depths=size_map[args.model_size]["depths"][::-1],
            num_heads=size_map[args.model_size]["num_heads"][::-1],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_snr_adapter=self.snr_adapter,
            use_token_pruner=self.token_pruner,
            use_channel_pruner=self.channel_pruner,
            module_hidden_ratio=1,
        )
