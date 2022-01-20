import torch
import torch.nn as nn
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
from torch.nn import ReLU, Sigmoid


#############
# UTILS     #
#############

def print_model(model):
    """
    Print model and parameters per layer to debug.
    @param model: torch.model
    """
    print("Model structure: ", model, "\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


##############
# 3D U-NET   #
##############

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm3d(out_channels),  # TODO: default parameters != tf parameters
            ReLU()
        )

    def forward(self, x):
        conv = self.block(x)
        return conv


class DownBlock(nn.Module):
    def __init__(self, in_channels, output_channels):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(in_channels, output_channels),
            ConvBlock(output_channels, output_channels)
        )
        self.max_pooling = MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        convs = self.convs(x)
        max_pool = self.max_pooling(convs)
        return convs, max_pool


class UpBlock(nn.Module):
    def __init__(self, in_channels, output_channels):
        super().__init__()
        self.deconv = ConvTranspose3d(in_channels, output_channels,
                                      kernel_size=2, stride=2, padding=0)
        self.convs = nn.Sequential(
            ConvBlock(in_channels, output_channels),
            ConvBlock(output_channels, output_channels)
        )

    def forward(self, skip, x):
        deconv = self.deconv(x)
        concat = torch.cat([deconv, skip], dim=1)
        convs = self.convs(concat)
        return convs


class UNet3D(nn.Module):
    def __init__(self, filters=8):
        super(UNet3D, self).__init__()

        # Encoding
        self.down_block_1 = DownBlock(4, filters)
        self.down_block_2 = DownBlock(filters, filters * 2)
        self.down_block_3 = DownBlock(filters * 2, filters * 4)

        self.bottleneck = nn.Sequential(
            ConvBlock(filters * 4, filters * 8),
            ConvBlock(filters * 8, filters * 8)
        )

        # Decoding
        self.up_block_1 = UpBlock(filters * 8, filters * 4)
        self.up_block_2 = UpBlock(filters * 4, filters * 2)
        self.up_block_3 = UpBlock(filters * 2, filters)

        self.output = nn.Sequential(
            Conv3d(in_channels=filters, out_channels=1, kernel_size=1,
                   padding=0),
            Sigmoid()
        )

    # noinspection PyCallingNonCallable
    def forward(self, x):
        skip_1, down_level_1 = self.down_block_1(x)
        skip_2, down_level_2 = self.down_block_2(down_level_1)
        skip_3, down_level_3 = self.down_block_3(down_level_2)

        level_4 = self.bottleneck(down_level_3)

        up_level_3 = self.up_block_1(skip_3, level_4)
        up_level_2 = self.up_block_2(skip_2, up_level_3)
        up_level_1 = self.up_block_3(skip_1, up_level_2)

        output = self.output(up_level_1)
        return output
