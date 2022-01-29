import torch.nn as nn

class Darknet(nn.Module):
    """ Backbone for YOLO 3D.
        Adapted from Darknet-53.
        Make sure the input image size can be devided by 16.
        
        Output 1/16, 1/8, 1/4 feature maps for later use.
    """

    def __init__(self, in_channels=1):
        super(Darknet, self).__init__()
        self.conv0 = nn.Sequential(ConvBlock(in_channels, 32, kernel_size=3), nResBlock(32, 4))
        self.features1 = nn.Sequential(DownSample(32, 64), nResBlock(64, 8), DownSample(64, 128), nResBlock(128, 8))#1/4
        self.features2 = nn.Sequential(DownSample(128, 256), nResBlock(256, 8))#1/8
        self.features3 = nn.Sequential(DownSample(256, 512), nResBlock(512, 8))#1/16
    
    def forward(self, x):
        x = self.conv0(x)
        x3 = self.features1(x)
        x2 = self.features2(x3)
        x1 = self.features3(x2)
        """
            Return 3 feature maps
            Note that x1 is the smallest map
        """
        return x1, x2, x3



class ConvBlock(nn.Module):
    """ Convolutional block.
        A sequence of convolution layer, batch normalization and leakyReLU.
        Always padding if needed.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        #3D version
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, kernel_size // 2), 
                                   nn.BatchNorm3d(out_channels), 
                                   nn.LeakyReLU(0.1))
    
    def forward(self, x):
        x = self.block(x)
        return x



class DownSample(nn.Module):
    """ Downsampling module.
        A convolution block with stride=2.
    """

    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)
    
    def forward(self, x):
        x = self.block(x)
        return x



class ResBlock(nn.Module):
    """ Residual module.
        Contains 2 convolutional block, the output channels of the first convolutional block is 1/2 of the input channels.
        Input channels and output channels are the same.
    """

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block1 = ConvBlock(in_channels, in_channels // 2, kernel_size=1)
        self.block2 = ConvBlock(in_channels // 2, in_channels, kernel_size=3)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out += x
        return out



class nResBlock(nn.Module):
    """ Pile up n residual modules above.
        All channels are the same.
    """

    def __init__(self, channels, n):
        super(nResBlock, self).__init__()
        blocklist = []
        for i in range(n):
            blocklist.append(ResBlock(channels))
        self.nres_module = nn.Sequential(*list([m for m in blocklist]))
    
    def forward(self, x):
        x = self.nres_module(x)
        return x
