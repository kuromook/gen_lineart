"""GAN Generator による推論スクリプト"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

IMAGE_SIZE = 480


class ConvNormReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )
    def forward(self, x):
        return F.relu(self.net(x))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvNormReLU(ch, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm  = nn.InstanceNorm2d(ch, affine=True)
    def forward(self, x):
        return x + self.norm(self.conv2(F.relu(self.conv1(x))))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class GANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(ConvNormReLU(1, 64),   ResBlock(64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(ConvNormReLU(64, 128),  ResBlock(128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(ConvNormReLU(128, 256), ResBlock(256))
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvNormReLU(256, 512), ResBlock(512))

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(ConvNormReLU(512, 256), ResBlock(256))
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(ConvNormReLU(256, 128), ResBlock(128))
        self.up1  = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec1 = nn.Sequential(ConvNormReLU(128, 64),  ResBlock(64))

        self.final = nn.Conv2d(64, 1, 1)
        self.spatial_attn1 = SpatialAttention()
        self.spatial_attn2 = SpatialAttention()
        self.spatial_attn3 = SpatialAttention()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  self.spatial_attn3(e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.spatial_attn2(e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.spatial_attn1(e1)], 1))
        return self.final(d1)


def run_inference(checkpoint, input_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = GANGenerator().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()

    img = Image.open(input_path).convert("L")
    tensor = TF.to_tensor(TF.resize(img, (IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0).to(device)

    with torch.no_grad():
        out = torch.sigmoid(G(tensor))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    TF.to_pil_image(out[0].cpu()).save(output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_gan/model_epoch_130.pth")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     default="results/gan_output.png")
    args = parser.parse_args()

    run_inference(args.checkpoint, args.input, args.output)
