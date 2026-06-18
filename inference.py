import argparse, torch, numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from models.generator import RRDBNet
from utils.metrics    import compute_psnr, compute_ssim
from utils.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--input",       required=True)
    p.add_argument("--output",      default="sr_output.png")
    p.add_argument("--hr_ref",      default=None)
    p.add_argument("--scale",       type=int, default=4)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--num_blocks",  type=int, default=23)
    return p.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = RRDBNet(args.in_channels, args.in_channels,
                num_blocks=args.num_blocks, scale_factor=args.scale).to(device)
    load_checkpoint(args.checkpoint, G)
    G.eval()

    lr = TF.to_tensor(Image.open(args.input).convert("L")).unsqueeze(0).to(device)
    sr = G(lr).clamp(0, 1).squeeze(0).cpu()

    arr = (sr.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, "L").save(args.output)
    print(f"Saved SR image → {args.output}")

    if args.hr_ref:
        hr = TF.to_tensor(Image.open(args.hr_ref).convert("L"))
        print(f"PSNR: {compute_psnr(sr, hr):.2f} dB")
        print(f"SSIM: {compute_ssim(sr, hr):.4f}")


if __name__ == "__main__":
    main()