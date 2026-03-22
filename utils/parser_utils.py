import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="SwinJSCC")
    parser.add_argument("--training", action="store_true", help="training or testing")
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        choices=[256, 512],
        help="training image size",
    )
    parser.add_argument(
        "--trainset",
        type=str,
        default="DIV2K",
        choices=["COCO", "DIV2K"],
        help="train dataset name",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="ffhq",
        choices=["Kodak", "CLIC21", "ffhq", "COCO"],
        help="specify the testset for HR models",
    )
    parser.add_argument(
        "--distortion-metric",
        type=str,
        default="MSE",
        choices=["MSE", "MS-SSIM"],
        help="evaluation metrics",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SwinJSCC_w/_SAandRA",
        choices=[
            "SwinJSCC_w/o_SAandRA",
            "SwinJSCC_w/_SA",
            "SwinJSCC_w/_RA",
            "SwinJSCC_w/_SAandRA",
        ],
        help="SwinJSCC model or SwinJSCC without channel ModNet or rate ModNet",
    )
    parser.add_argument(
        "--channel-type",
        type=str,
        default="awgn",
        choices=["awgn", "rayleigh"],
        help="wireless channel model, awgn or rayleigh",
    )
    parser.add_argument("--C", type=str, default="96", help="bottleneck dimension")
    parser.add_argument(
        "--multiple-snr", type=str, default="10", help="random or fixed snr"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["small", "base", "large", "baseline"],
        help="SwinJSCC model size",
    )
    parser.add_argument("--attractor", action="store_true", help="add attractor module")
    parser.add_argument(
        "--encoder-adapter", action="store_true", help="add encoder_adapter module"
    )
    parser.add_argument(
        "--decoder-adapter", action="store_true", help="add decoder_adapter module"
    )
    parser.add_argument("--sr", action="store_true", help="add sr module")
    parser.add_argument(
        "--amp", action="store_true", help="enable torch.cuda.amp mixed precision"
    )
    return parser
