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


class DarkenOnlyCleanupGenerator(nn.Module):
    """Shallow refiner like ResidualCleanupGenerator, but the correction can only
    add ink (raise the logit) around the atari base, never remove it.

    Rationale: the bidirectional (tanh) correction can freely erase atari ink as
    well as add it, which lets the network paint soft partial-erasure/partial-
    addition gray everywhere instead of committing to a binary decision. A
    one-sided, smaller-range correction forces the model to trust the atari
    base except where it confidently wants to add a missed line.
    """

    def __init__(self, in_channels=2, out_channels=1, channels=48, blocks=5, max_delta=1.5):
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
        final = nn.Conv2d(channels, out_channels, 3, padding=1)
        # Start near zero addition so early training matches the atari base
        # closely, rather than starting at the sigmoid(0)=0.5 midpoint.
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, -4.0)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        delta = torch.sigmoid(self.net(x)) * self.max_delta
        if x.shape[1] < 2:
            return delta
        aux_ink = (1.0 - x[:, 1:2]).clamp(1e-4, 1.0 - 1e-4)
        aux_logits = torch.logit(aux_ink)
        return aux_logits + delta


def aux_channel_logits(x):
    """Convert the aux (atari) input channel back to a logit map, so a
    refiner can predict a small bounded correction around it instead of
    reconstructing ink density from scratch. Returns 0.0 (no anchor) if x has
    no aux channel."""
    if x.shape[1] < 2:
        return 0.0
    aux_ink = (1.0 - x[:, 1:2]).clamp(1e-4, 1.0 - 1e-4)
    return torch.logit(aux_ink)


class DualHeadRefinerGenerator(nn.Module):
    """Direction 6 (retried with residual anchoring): separate confidence
    (skeleton/centerline) from thickness (normal full-width ink) instead of
    asking one output tensor to be both.

    The first attempt reconstructed ink_logits directly from a fresh trunk
    with no anchor to the aux (atari) channel, and converged to a chronically
    under-inked output after only 3 epochs -- that training budget was
    calibrated for residual-anchored models (ResidualCleanupGenerator /
    DarkenOnlyCleanupGenerator), not from-scratch ones. This version keeps
    the dual-head idea -- a separate skeleton/centerline confidence head,
    supervised via `--skeleton-weight`, composed onto the ink output -- but
    makes `ink_head` predict a small bounded correction around the aux base
    (like every other adopted model in this family), and zero/negative-bias
    -inits both heads so the initial output matches the aux base closely.

    In training mode, forward returns (final_logits, skeleton_logits) so the
    training loop can supervise the skeleton head directly. In eval mode
    (inference), forward returns only final_logits, matching every other
    generator's inference contract.
    """

    def __init__(self, in_channels=2, out_channels=1, channels=48, blocks=5, max_delta=4.0, skeleton_gain=1.5):
        super().__init__()
        self.max_delta = max_delta
        self.skeleton_gain = skeleton_gain
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
        self.trunk = nn.Sequential(*layers)
        self.ink_head = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.skeleton_head = nn.Conv2d(channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.ink_head.weight)
        nn.init.zeros_(self.ink_head.bias)
        nn.init.zeros_(self.skeleton_head.weight)
        nn.init.constant_(self.skeleton_head.bias, -4.0)

    def forward(self, x):
        feat = self.trunk(x)
        ink_delta = torch.tanh(self.ink_head(feat)) * self.max_delta
        skeleton_logits = self.skeleton_head(feat)
        final_logits = aux_channel_logits(x) + ink_delta + self.skeleton_gain * torch.sigmoid(skeleton_logits)
        if self.training:
            return final_logits, skeleton_logits
        return final_logits


class HedUNetGenerator(nn.Module):
    """Direction 8 (retried with residual anchoring): HED/DexiNed-style 2ch
    U-Net refiner with multi-scale side outputs.

    Same encoder/decoder as the plain `refiner_unet` (`unet` model_name with
    `aux_dir`), plus two extra 1x1-conv heads reading the decoder's 1/4- and
    1/2-resolution feature maps directly, each supervised (via
    `--side-weight`) against the GT line target downsampled to that head's
    native resolution -- the standard HED "side output" trick, adapted from
    natural-image edge detection to line art. The side heads have zero effect
    on the final output itself; they only shape gradients into the shared
    decoder trunk during training.

    The first attempt had `out_conv` reconstruct final_logits directly from
    the decoder, with no anchor to the aux (atari) channel, and converged to
    a chronically under-inked output after only 3 epochs -- that training
    budget was calibrated for residual-anchored models
    (ResidualCleanupGenerator / DarkenOnlyCleanupGenerator), not from-scratch
    ones. This version makes `out_conv` predict a small bounded correction
    around the aux base instead, and zero-inits it so the initial output
    matches the aux base closely, same as the dual-head retry.

    In training mode, forward returns (final_logits, [side_logits_1_4,
    side_logits_1_2]). In eval mode, forward returns only final_logits,
    matching every other generator's inference contract.
    """

    def __init__(self, in_channels=2, out_channels=1, max_delta=4.0):
        super().__init__()
        self.max_delta = max_delta
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
        self.side3 = nn.Conv2d(256, out_channels, 1)
        self.side2 = nn.Conv2d(128, out_channels, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        delta = torch.tanh(self.out_conv(d1)) * self.max_delta
        final_logits = aux_channel_logits(x) + delta
        if self.training:
            return final_logits, [self.side3(d3), self.side2(d2)]
        return final_logits


class SelfAttention2d(nn.Module):
    """Lightweight non-local self-attention (SAGAN-style), meant to run only
    on a small bottleneck feature map (see AttentionUNetGenerator) since the
    attention matrix is O((h*w)^2). `gamma` is zero-initialized so the block
    starts as an identity no-op and only starts mixing distant positions
    together once training pushes it away from zero."""

    def __init__(self, channels, key_channels=None):
        super().__init__()
        key_channels = key_channels or max(channels // 8, 8)
        self.query = nn.Conv2d(channels, key_channels, 1)
        self.key = nn.Conv2d(channels, key_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k) / (k.shape[1] ** 0.5), dim=-1)
        v = self.value(x).view(b, c, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.gamma * out


class AttentionUNetGenerator(nn.Module):
    """Direction 9: lightweight bottleneck self-attention added to the plain
    2ch U-Net refiner (`unet` model_name + aux_dir), so the refiner can mix
    information between distant pixels -- e.g. reconcile a hair strand on one
    side of the face with strokes on the other -- instead of being limited to
    the U-Net's local conv receptive field. Attention runs only at the
    bottleneck (60x60 for a 480px input after 3 stride-2 pools), keeping the
    O(N^2) attention matrix small.

    Residual-anchored to the aux (atari) input from the start (out =
    aux_logits + bounded correction), applying this session's Direction 6/8
    lesson directly rather than re-discovering it: an unanchored from-scratch
    output needs far more epochs than this project's short-survey budget to
    reach a usable ink density. `out_conv` is zero-inited and the attention
    block's `gamma` starts at zero, so the whole model starts at exactly the
    aux baseline, same as the fixed DualHeadRefinerGenerator/HedUNetGenerator.

    Forward always returns a single tensor (no auxiliary heads -- attention
    has no separate supervision target), so no training-loop changes are
    needed beyond registering the model name.
    """

    def __init__(self, in_channels=2, out_channels=1, max_delta=4.0):
        super().__init__()
        self.max_delta = max_delta
        base = UNetGenerator(in_channels=in_channels, out_channels=out_channels)
        self.enc1 = base.enc1
        self.pool1 = base.pool1
        self.enc2 = base.enc2
        self.pool2 = base.pool2
        self.enc3 = base.enc3
        self.pool3 = base.pool3
        self.bottleneck = base.bottleneck
        self.attn = SelfAttention2d(512)
        self.up3 = base.up3
        self.dec3 = base.dec3
        self.up2 = base.up2
        self.dec2 = base.dec2
        self.up1 = base.up1
        self.dec1 = base.dec1
        self.out_conv = base.out_conv
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        b = self.attn(b)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        delta = torch.tanh(self.out_conv(d1)) * self.max_delta
        return aux_channel_logits(x) + delta


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
    if model_name == "cleanupdark":
        return DarkenOnlyCleanupGenerator(in_channels=in_channels)
    if model_name == "dualhead":
        return DualHeadRefinerGenerator(in_channels=in_channels)
    if model_name == "hed":
        return HedUNetGenerator(in_channels=in_channels)
    if model_name == "attn":
        return AttentionUNetGenerator(in_channels=in_channels)
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
