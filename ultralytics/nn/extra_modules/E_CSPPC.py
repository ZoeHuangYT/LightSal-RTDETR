import torch
import torch.nn as nn
__all__ = ['E_CSPPC']


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # self.se = CBAM(c2)
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class GatedConv(nn.Module):
    """A convolutional layer with a gating mechanism."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)  # Gating layer

    def forward(self, x):
        conv_out = self.conv(x)
        gate_out = torch.sigmoid(self.gate(x))  # Sigmoid activation for gating
        return conv_out * gate_out  # Element-wise multiplication with gate


class Partial_Gated_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_gated_conv3 = GatedConv(self.dim_conv3, self.dim_conv3)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_gated_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_gated_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class CSPPC_Bottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.DualPConv = nn.Sequential(Partial_Gated_conv3(dim, n_div=4, forward='split_cat'), Partial_Gated_conv3(dim, n_div=4, forward='split_cat'))

    def forward(self, x):
        return self.DualPConv(x)

class E_CSPPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CSPPC_Bottleneck(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))




if __name__ == "__main__":
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)
    model = E_CSPPC(64, 128)
    out = model(image)
    print(out.size())
