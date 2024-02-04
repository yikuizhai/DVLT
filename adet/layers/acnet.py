import torch.nn as nn
import torch.nn.init as init
import torch

class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
                 use_affine=True, reduce_gamma=False, gamma_init=None ):
        super(ACBlock, self).__init__()

        self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(kernel_size, kernel_size), stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=False,
                                     padding_mode=padding_mode)
        self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)


        if padding - kernel_size // 2 >= 0:
            #   Common use case. E.g., k=3, p=1 or k=5, p=2
            self.crop = 0
            #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
            hor_padding = [padding - kernel_size // 2, padding]
            ver_padding = [padding, padding - kernel_size // 2]
        else:
            #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
            #   Since nn.Conv2d does not support negative padding, we implement it manually
            self.crop = kernel_size // 2 - padding
            hor_padding = [0, padding]
            ver_padding = [padding, 0]

        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                  stride=stride,
                                  padding=ver_padding, dilation=dilation, groups=groups, bias=False,
                                  padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                  stride=stride,
                                  padding=hor_padding, dilation=dilation, groups=groups, bias=False,
                                  padding_mode=padding_mode)
        self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
        self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

        if reduce_gamma:
            self.init_gamma(1.0 / 3)

        if gamma_init is not None:
            assert not reduce_gamma
            self.init_gamma(gamma_init)

    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)
        print('init gamma of square, ver and hor as ', gamma_value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')

    def forward(self, input):

        square_outputs = self.square_conv(input)
        square_outputs = self.square_bn(square_outputs)
        if self.crop > 0:
            ver_input = input[:, :, :, self.crop:-self.crop]
            hor_input = input[:, :, self.crop:-self.crop, :]
        else:
            ver_input = input
            hor_input = input
        vertical_outputs = self.ver_conv(ver_input)
        vertical_outputs = self.ver_bn(vertical_outputs)
        horizontal_outputs = self.hor_conv(hor_input)
        horizontal_outputs = self.hor_bn(horizontal_outputs)
        result = square_outputs + vertical_outputs + horizontal_outputs
        return result