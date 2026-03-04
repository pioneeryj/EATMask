import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True


class STUNet(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in
              range(depth[0] - 1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                                                stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                                  *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                                    for _ in range(depth[d] - 1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                                  *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                  self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            skip = skips[-(u + 1)]
            # ensure spatial dims match before concatenation by center-cropping both to common min size
            x, skip = _center_crop_to_common(x, skip)
            x = torch.cat((x, skip), dim=1)
            x = self.conv_blocks_localization[u](x)
            if self.decoder.deep_supervision:
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            # compute only final segmentation head when deep supervision is disabled
            return self.final_nonlin(self.seg_outputs[-1](x))


class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


def _match_spatial_size(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Make `src` have the same D, H, W as `ref` via symmetric center crop/pad.
    Expects tensors of shape (N, C, D, H, W).
    """
    sd, sh, sw = src.shape[2], src.shape[3], src.shape[4]
    rd, rh, rw = ref.shape[2], ref.shape[3], ref.shape[4]

    # Center crop when src is larger
    if sd > rd:
        start = (sd - rd) // 2
        src = src[:, :, start:start + rd, :, :]
        sd = rd
    if sh > rh:
        start = (sh - rh) // 2
        src = src[:, :, :, start:start + rh, :]
        sh = rh
    if sw > rw:
        start = (sw - rw) // 2
        src = src[:, :, :, :, start:start + rw]
        sw = rw

    # Symmetric zero-pad when src is smaller
    pd_d = max(rd - sd, 0)
    pd_h = max(rh - sh, 0)
    pd_w = max(rw - sw, 0)

    if pd_d or pd_h or pd_w:
        d_left = pd_d // 2
        d_right = pd_d - d_left
        h_top = pd_h // 2
        h_bottom = pd_h - h_top
        w_left = pd_w // 2
        w_right = pd_w - w_left
        # pad order for 3D: (w_left, w_right, h_top, h_bottom, d_left, d_right)
        src = nn.functional.pad(src, (w_left, w_right, h_top, h_bottom, d_left, d_right))

    return src


def _center_crop_to_common(a: torch.Tensor, b: torch.Tensor):
    """
    Center-crop both tensors to the common minimum spatial size.
    Expects shape (N, C, D, H, W). Returns cropped (a, b).
    """
    ad, ah, aw = a.shape[2], a.shape[3], a.shape[4]
    bd, bh, bw = b.shape[2], b.shape[3], b.shape[4]

    td, th, tw = min(ad, bd), min(ah, bh), min(aw, bw)

    def crop_center(x, td, th, tw):
        xd, xh, xw = x.shape[2], x.shape[3], x.shape[4]
        sd = (xd - td) // 2
        sh = (xh - th) // 2
        sw = (xw - tw) // 2
        return x[:, :, sd:sd+td, sh:sh+th, sw:sw+tw]

    a_c = crop_center(a, td, th, tw) if (ad != td or ah != th or aw != tw) else a
    b_c = crop_center(b, td, th, tw) if (bd != td or bh != th or bw != tw) else b

    return a_c, b_c