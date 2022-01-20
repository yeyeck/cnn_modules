import torch
from torch import nn
from typing import  Union, Tuple
from typing import Callable, Any, Optional, List
from torchvision import models



size_2t = Union[int, Tuple[int, int]]




class Conv2dWithBNA(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, 
                    kernel_size: size_2t=1,
                    stride: int = 1,
                    padding: size_2t = 0,
                    dilation: int = 1,
                    groups: int = 1,
                    bias: bool = True,
                    norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                    activation: Optional[Callable[..., nn.Module]] = nn.ReLU):
        super().__init__(
            nn.Conv2d(in_channels, out_channels,kernel_size, stride, padding, dilation, groups, bias),
            norm_layer(out_channels),
            activation()
        )



class InceptionA(nn.Module):
    '''
    Implementation of << Going deeper with convolutions >> Figure 2(a): Inception module, native version
    see https://arxiv.org/pdf/1409.4842.pdf for details
    '''
    def __init__(self, in_channels: int, 
                    channels_1x1: int, 
                    channels_3x3: int, 
                    channels_5x5: int,
                    activation: Optional[Callable[..., nn.Module]] = nn.ReLU):
        super().__init__()
        self.conv1x1 = Conv2dWithBNA(in_channels, channels_1x1, 1, 1, 0, activation=activation)
        self.conv3x3 = Conv2dWithBNA(in_channels, channels_3x3, 3, 1, 1, activation=activation)
        self.conv5x5 = Conv2dWithBNA(in_channels, channels_5x5, 5, 1, 2, activation=activation)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        o1 = self.conv1x1(x)
        o2 = self.conv3x3(x)
        o3 = self.conv5x5(x)
        o4 = self.pooling(x)
        x = torch.cat((o1, o2, o3, o4), dim=1)
        return x


class InceptionB(nn.Module):
    '''
    Implementation of << Going deeper with convolutions >> Figure 2(b):  Inception module with dimension reductions
    see https://arxiv.org/pdf/1409.4842.pdf for details
    '''
    def __init__(self, in_channels: int, 
                    channels_1x1: int, 
                    channels_3x3: int, 
                    channels_5x5: int,
                    channels_polling: int,
                    channel_scale: float,
                    activation: Optional[Callable[..., nn.Module]] = nn.ReLU):
        super().__init__()
        scaled_channel = int(in_channels * channel_scale)
        self.branch1 = Conv2dWithBNA(in_channels, channels_1x1, 1, 1, 0, activation=activation)
        self.branch3 = nn.Sequential(
            Conv2dWithBNA(in_channels, scaled_channel, 1, 1, 0, activation=activation),
            Conv2dWithBNA(scaled_channel, channels_3x3, 3, 1, 1, activation=activation)
        )
        self.branch5 = nn.Sequential(
            Conv2dWithBNA(in_channels, scaled_channel, 1, 1, 0, activation=activation),
            Conv2dWithBNA(scaled_channel, channels_5x5, 5, 1, 2, activation=activation)
        )
        self.branch_pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2dWithBNA(in_channels, channels_polling, 1, 1, 0, activation=activation)
        )

    def forward(self, x):
        o1 = self.branch1(x)
        o2 = self.branch3(x)
        o3 = self.branch5(x)
        o4 = self.branch_pooling(x)
        x = torch.cat((o1, o2, o3, o4), dim=1)
        return x

class InceptionC(nn.Sequential):
    '''
    Implementation of << Rethinking the Inception Architecture for Computer Vision >> Figure 5: Inception modules where each 5x5 convolution is replaced by two 3x3 convolution
    see https://arxiv.org/pdf/1512.00567.pdf for details
    '''
    def __init__(self, in_channels: int, 
                    channels_1x1: int, 
                    channels_3x3: int, 
                    channels_5x5: int,
                    channels_polling: int,
                    channel_scale: float,
                    activation: Optional[Callable[..., nn.Module]] = nn.ReLU):
        super().__init__()
        scaled_channel = int(in_channels * channel_scale)
        self.branch1 = Conv2dWithBNA(in_channels, channels_1x1, 1, 1, 0, activation=activation)
        self.branch3 = nn.Sequential(
            Conv2dWithBNA(in_channels, scaled_channel, 1, 1, 0, activation=activation),
            Conv2dWithBNA(scaled_channel, channels_3x3, 3, 1, 1, activation=activation)
        )
        self.branch5 = nn.Sequential(
            Conv2dWithBNA(in_channels, scaled_channel, 1, 1, 0, activation=activation),
            Conv2dWithBNA(scaled_channel, scaled_channel, 3, 1, 1, activation=activation),
            Conv2dWithBNA(scaled_channel, channels_5x5, 3, 1, 1, activation=activation)
        )
        self.branch_pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2dWithBNA(in_channels, channels_polling, 1, 1, 0, activation=activation)
        )

    def forward(self, x):
        o1 = self.branch1(x)
        o2 = self.branch3(x)
        o3 = self.branch5(x)
        o4 = self.branch_pooling(x)
        x = torch.cat((o1, o2, o3, o4), dim=1)
        return x

class InceptionD(nn.Sequential):
    '''
    Implementation of << Rethinking the Inception Architecture for Computer Vision >> Figure 6: Inception modules after the factorization of the nxn convolutions
    see https://arxiv.org/pdf/1512.00567.pdf for details
    '''

    def __init__(self, in_channels: int, 
                    channels_1x1: int, 
                    channels_3x3: int, 
                    channels_5x5: int,
                    channels_polling: int,
                    channel_scale: float,
                    kernel_size: int = 3,
                    activation: Optional[Callable[..., nn.Module]] = nn.ReLU):
        super().__init__()
        scaled_channel = int(in_channels * channel_scale)
        padding = (kernel_size - 1) // 2
        self.branch1 = Conv2dWithBNA(in_channels, channels_1x1, 1, 1, 0)
        self.branch3 = nn.Sequential(
            Conv2dWithBNA(in_channels, scaled_channel, 1, 1, 0, activation=activation),
            Conv2dWithBNA(scaled_channel, scaled_channel, (1, kernel_size), 1, (0, padding), activation=activation),
            Conv2dWithBNA(scaled_channel, channels_3x3, (kernel_size, 1), 1, (padding, 0), activation=activation)
        )
        self.branch5 = nn.Sequential(
            Conv2dWithBNA(in_channels, scaled_channel, 1, 1, 0),
            Conv2dWithBNA(scaled_channel, scaled_channel, (1, kernel_size), 1, (0, padding), activation=activation),
            Conv2dWithBNA(scaled_channel, scaled_channel, (kernel_size, 1), 1, (padding, 0), activation=activation),
            Conv2dWithBNA(scaled_channel, scaled_channel, (1, kernel_size), 1, (0, padding), activation=activation),
            Conv2dWithBNA(scaled_channel, channels_5x5, (kernel_size, 1), 1, (padding, 0), activation=activation)
        )
        self.branch_pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2dWithBNA(in_channels, channels_polling, 1, 1, 0, activation=activation)
        )

    def forward(self, x):
        o1 = self.branch1(x)
        o2 = self.branch3(x)
        o3 = self.branch5(x)
        o4 = self.branch_pooling(x)
        x = torch.cat((o1, o2, o3, o4), dim=1)
        return x


class InceptionE(nn.Sequential):
    '''
    Implementation of << Rethinking the Inception Architecture for Computer Vision >> Figure 7: Inception modules with expanded the filter bank outputs
    see https://arxiv.org/pdf/1512.00567.pdf for details
    '''
    def __init__(self, in_channels: int, 
                    channels_1x1: int, 
                    channels_3x3: int, 
                    channels_5x5: int,
                    channels_polling: int,
                    channel_scale: float,
                    activation: Optional[Callable[..., nn.Module]] = nn.ReLU):
        super().__init__()
        scaled_channel = int(in_channels * channel_scale)
        self.branch1 = Conv2dWithBNA(in_channels, channels_1x1, 1, 1, 0, activation=activation)
        self.branch3_1x1 = Conv2dWithBNA(in_channels, scaled_channel, 1, 1, 0, activation=activation)
        self.branch3_1x3 = Conv2dWithBNA(scaled_channel, channels_3x3, (1, 3), 1, (0, 1), activation=activation)
        self.branch3_3x1 = Conv2dWithBNA(scaled_channel, channels_3x3, (3, 1), 1, (1, 0), activation=activation)

        self.branch5_3x3 = nn.Sequential(
            Conv2dWithBNA(in_channels, scaled_channel, 1, 1, 0, activation=activation),
            Conv2dWithBNA(scaled_channel, scaled_channel, 3, 1, 1, activation=activation)
        )
        self.branch5_1x3 = Conv2dWithBNA(scaled_channel, channels_5x5, (1, 3), 1, (0, 1), activation=activation)
        self.branch5_3x1 = Conv2dWithBNA(scaled_channel, channels_5x5, (3, 1), 1, (1, 0), activation=activation)

        self.branch_pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2dWithBNA(in_channels, channels_polling, 1, 1, 0, activation=activation)
        )

    def forward(self, x):
        o1 = self.branch1(x)
        
        o2 = self.branch3_1x1(x)
        o21 = self.branch3_1x3(o2)
        o22 = self.branch3_3x1(o2)

        o3 = self.branch5_3x3(x)
        o31 = self.branch5_1x3(o3)
        o32 = self.branch5_3x1(o3)

        o4 = self.branch_pooling(x)
        x = torch.cat((o1, o21, o22, o31, o32, o4), dim=1)
        return x


if __name__ == '__main__':
    data = torch.randn(1, 64, 256, 256)
    from thop import profile, clever_format
    from onnxsim import simplify
    import onnx

    conv = Conv2dWithBNA(64, 256)
    inceptiona = InceptionA(64, 64, 64, 64)
    inceptionb = InceptionB(64, 48, 96, 96, 48, 0.5)
    inceptionc = InceptionC(64, 48, 96, 96, 48, 0.5)
    inceptiond = InceptionC(64, 48, 96, 96, 48, 0.5)
    inceptione = InceptionE(64, 32, 64, 64, 32, 0.5)
    modules = [conv, inceptiona, inceptionb, inceptionc, inceptiond, inceptione]

    names = ['conv', 'inceptiona', 'inceptionb', 'inceptionc', 'inceptiond', 'inceptione']
    for module, name in zip(modules, names):
        pth = name + '.onnx'
        torch.onnx.export(module, data, pth)
        model = onnx.load_model(pth)
        model_simp, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, pth)
