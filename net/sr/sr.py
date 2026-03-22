import torch
import torch.nn as nn
from .dmnet_arch import DMNet
import torchvision.transforms as T
from PIL import Image
import os


class SRNet(nn.Module):
    """
    Plug-and-play SR module for JSCC decoder output.
    """

    def __init__(self, ckpt_path):
        super().__init__()

        # ---- build DMNet directly ----
        self.net = DMNet(
            upscale=2,
            in_chans=3,
            dim=48,
            groups=3,
            blocks=3,
            buildblock_type="Wave",
            restormer_num_heads=8,
            restormer_ffn_type="GDFN",
            restormer_ffn_expansion_factor=2.0,
            tlc_flag=False,  # IMPORTANT: disable TLC for batch inference
            tlc_kernel=64,
            activation="relu",
            body_norm=False,
            img_range=1.0,
            upsampler="pixelshuffledirect",
        )

        state = torch.load(ckpt_path)
        if "params" in state:
            state = state["params"]

        self.net.load_state_dict(state, strict=True)

    def forward(self, x):
        """
        x: JSCC reconstructed image, shape [B,3,H,W], range [0,1]
        """
        return torch.clamp(self.net(x), 0.0, 1.0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. load image ----
    img_path = "./test/recon_0_0_1_192.png"  # CHANGE THIS
    img = Image.open(img_path).convert("RGB")
    x = T.ToTensor()(img).unsqueeze(0).to(device)  # [1,3,H,W], [0,1]

    # ---- 2. load SR model ----
    # Replace this with YOUR SR model
    sr_model = SRNet(ckpt_path="./pretrained/dmnet_x2.pth").to(device)

    sr_model.eval()

    # ---- 3. run SR ----
    with torch.no_grad():
        y = sr_model(x)

    # ---- 4. save output ----
    y = y.clamp(0, 1)[0].cpu()
    out_img = T.ToPILImage()(y)
    out_img.save("./test/output_sr.png")

    print("SR done. Saved to output_sr.png")
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()
