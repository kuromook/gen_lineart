import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from lineart.unetgenerator import UNetGenerator


class ScaledSkipUNet(nn.Module):
    """UNetGenerator-compatible model with attenuated decoder skip features."""

    def __init__(self, in_channels=1, out_channels=1, skip_scale=0.5):
        super().__init__()
        self.skip_scale = skip_scale
        base = UNetGenerator(in_channels=in_channels, out_channels=out_channels)
        self.enc1 = base.enc1
        self.pool1 = base.pool1
        self.enc2 = base.enc2
        self.pool2 = base.pool2
        self.enc3 = base.enc3
        self.pool3 = base.pool3
        self.bottleneck = base.bottleneck
        self.up3 = base.up3
        self.dec3 = base.dec3
        self.up2 = base.up2
        self.dec2 = base.dec2
        self.up1 = base.up1
        self.dec1 = base.dec1
        self.out_conv = base.out_conv

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3 * self.skip_scale], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2 * self.skip_scale], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1 * self.skip_scale], dim=1))
        return self.out_conv(d1)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, transpose=False):
        super().__init__()
        if transpose:
            conv = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            )
        else:
            conv = nn.Conv2d(
                in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=False
            )
        self.net = nn.Sequential(
            conv,
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.net(x)


class ResnetGenerator(nn.Module):
    """Encoder-residual-decoder generator without direct high-res skip copies."""

    def __init__(self, in_channels=1, out_channels=1, base_channels=64, blocks=6):
        super().__init__()
        c = base_channels
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, c, 7, padding=0, bias=False),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            ConvNormAct(c, c * 2, kernel_size=3, stride=2, padding=1),
            ConvNormAct(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
        )
        self.body = nn.Sequential(*[ResnetBlock(c * 4) for _ in range(blocks)])
        self.up = nn.Sequential(
            ConvNormAct(c * 4, c * 2, transpose=True),
            ConvNormAct(c * 2, c, transpose=True),
        )
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, out_channels, 7, padding=0),
        )

    def forward(self, x):
        return self.tail(self.up(self.body(self.down(self.head(x)))))


class ResidualCleanupGenerator(nn.Module):
    """Shallow refiner that predicts a bounded correction around the atari input."""

    def __init__(self, in_channels=2, out_channels=1, channels=48, blocks=5, max_delta=4.0):
        super().__init__()
        self.max_delta = max_delta
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(channels, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        delta = torch.tanh(self.net(x)) * self.max_delta
        if x.shape[1] < 2:
            return delta
        aux_ink = (1.0 - x[:, 1:2]).clamp(1e-4, 1.0 - 1e-4)
        aux_logits = torch.logit(aux_ink)
        return aux_logits + delta


class MaskCleanupGenerator(nn.Module):
    """Shallow cleanup model that uses atari as context, not as the output base."""

    def __init__(self, in_channels=2, out_channels=1, channels=48, blocks=5):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(channels, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlowMaskCleanupGenerator(nn.Module):
    """Use the auxiliary channel as a soft line-flow gate for local corrections."""

    def __init__(self, in_channels=2, out_channels=1, channels=48, blocks=5, max_delta=4.0):
        super().__init__()
        self.max_delta = max_delta
        self.base = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(channels, out_channels, 3, padding=1))
        self.delta = nn.Sequential(*layers)

    def forward(self, x):
        rough = x[:, :1]
        base_logits = self.base(rough)
        if x.shape[1] < 2:
            return base_logits
        aux_ink = (1.0 - x[:, 1:2]).clamp(0.0, 1.0)
        gate = F.max_pool2d(aux_ink, kernel_size=7, stride=1, padding=3)
        gate = gate * gate * (3.0 - 2.0 * gate)
        delta = torch.tanh(self.delta(x)) * self.max_delta
        return base_logits + gate * delta


class FlowMaskUNetGenerator(nn.Module):
    """U-Net base with flow-mask-gated corrections from rough + aux."""

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        max_delta=3.0,
        gate_floor=0.10,
        gate_kernel=9,
    ):
        super().__init__()
        self.max_delta = max_delta
        self.gate_floor = gate_floor
        self.gate_kernel = gate_kernel
        self.base = UNetGenerator(in_channels=1, out_channels=out_channels)
        self.delta = UNetGenerator(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        rough = x[:, :1]
        base_logits = self.base(rough)
        if x.shape[1] < 2:
            return base_logits
        aux_ink = (1.0 - x[:, 1:2]).clamp(0.0, 1.0)
        gate = F.max_pool2d(
            aux_ink,
            kernel_size=self.gate_kernel,
            stride=1,
            padding=self.gate_kernel // 2,
        )
        gate = gate * gate * (3.0 - 2.0 * gate)
        gate = self.gate_floor + (1.0 - self.gate_floor) * gate
        delta = torch.tanh(self.delta(x)) * self.max_delta
        return base_logits + gate * delta


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()
        c = base_channels
        self.layers = nn.ModuleList(
            [
            nn.Conv2d(in_channels, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, 1, 4, stride=1, padding=1),
            ]
        )

    def forward(self, rough, line_ink, return_features=False):
        x = torch.cat([rough, line_ink], dim=1)
        features = []
        for layer in self.layers:
            x = layer(x)
            if return_features and isinstance(layer, (nn.LeakyReLU, nn.InstanceNorm2d)):
                features.append(x)
        if return_features:
            return x, features
        return x


class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self, scales=(1.0, 0.5), in_channels=2, base_channels=64):
        super().__init__()
        self.scales = scales
        self.discriminators = nn.ModuleList(
            [PatchDiscriminator(in_channels=in_channels, base_channels=base_channels) for _ in scales]
        )

    def _scale(self, x, scale):
        if scale == 1.0:
            return x
        return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, rough, line_ink, return_features=False):
        outputs = []
        features = []
        for scale, discriminator in zip(self.scales, self.discriminators):
            rough_s = self._scale(rough, scale)
            line_s = self._scale(line_ink, scale)
            if return_features:
                score, feats = discriminator(rough_s, line_s, return_features=True)
                outputs.append(score)
                features.append(feats)
            else:
                outputs.append(discriminator(rough_s, line_s))
        if return_features:
            return outputs, features
        return outputs


def parse_skip_scale(model_name):
    match = re.fullmatch(r"unet_skip([0-9]+)", model_name)
    if not match:
        return 1.0
    return int(match.group(1)) / 100.0


def build_generator(model_name, in_channels=1):
    if model_name == "unet":
        return UNetGenerator(in_channels=in_channels, out_channels=1)
    if model_name.startswith("unet_skip"):
        return ScaledSkipUNet(in_channels=in_channels, skip_scale=parse_skip_scale(model_name))
    if model_name == "resnet":
        return ResnetGenerator(in_channels=in_channels)
    if model_name == "cleanup":
        return ResidualCleanupGenerator(in_channels=in_channels)
    if model_name == "maskcleanup":
        return MaskCleanupGenerator(in_channels=in_channels)
    if model_name == "flowmaskcleanup":
        return FlowMaskCleanupGenerator(in_channels=in_channels)
    if model_name == "flowmaskunet":
        return FlowMaskUNetGenerator(in_channels=in_channels)
    if model_name == "softflowmaskunet":
        return FlowMaskUNetGenerator(
            in_channels=in_channels,
            max_delta=2.0,
            gate_floor=0.35,
            gate_kernel=13,
        )
    raise ValueError(f"unknown model: {model_name}")


def load_generator_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "G_state" in checkpoint:
        model_name = checkpoint.get("model_name", "unet")
        in_channels = checkpoint.get("in_channels", 1)
        model = build_generator(model_name, in_channels=in_channels).to(device)
        model.load_state_dict(checkpoint["G_state"])
        return model, model_name
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_name = checkpoint.get("model_name", "unet")
        in_channels = checkpoint.get("in_channels", 1)
        model = build_generator(model_name, in_channels=in_channels).to(device)
        model.load_state_dict(checkpoint["model_state"])
        return model, model_name

    model = UNetGenerator(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(checkpoint)
    return model, "unet"
