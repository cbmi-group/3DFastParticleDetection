import torch
import torch.nn as nn
from .Darknet import Darknet, ConvBlock

class YOLO3D(nn.Module):
    """ YOLO3D full network.
    
        Out_channels = 5(x, y, z, e, conf) + n(class_num).
    """

    def __init__(self, in_channels=1, out_channels=6):
        super(YOLO3D, self).__init__()
        self.darknet = Darknet(in_channels)

        self.conv1 = nn.Sequential(ConvBlock(512, 256, kernel_size=1), ConvBlock(256, 512, kernel_size=3), 
                                   ConvBlock(512, 256, kernel_size=1), ConvBlock(256, 512, kernel_size=3), 
                                   ConvBlock(512, 256, kernel_size=1))
        
        self.upsample1 = nn.Sequential(ConvBlock(256, 128, kernel_size=1), UpSample())

        self.conv2 = nn.Sequential(ConvBlock(256 + 128, 128, kernel_size=1), ConvBlock(128, 256, kernel_size=3), 
                                   ConvBlock(256, 128, kernel_size=1), ConvBlock(128, 256, kernel_size=3), 
                                   ConvBlock(256, 128, kernel_size=1))
        
        self.upsample2 = nn.Sequential(ConvBlock(128, 64, kernel_size=1), UpSample())

        self.conv3 = nn.Sequential(ConvBlock(128 + 64, 64, kernel_size=1), ConvBlock(64, 128, kernel_size=3), 
                                   ConvBlock(128, 64, kernel_size=1), ConvBlock(64, 128, kernel_size=3), 
                                   ConvBlock(128, 64, kernel_size=1))

        self.output_layer1 = nn.Sequential(ConvBlock(256, 512, kernel_size=3), ConvBlock(512, out_channels, kernel_size=1))

        self.output_layer2 = nn.Sequential(ConvBlock(128, 256, kernel_size=3), ConvBlock(256, out_channels, kernel_size=1))

        self.output_layer3 = nn.Sequential(ConvBlock(64, 128, kernel_size=3), ConvBlock(128, out_channels, kernel_size=1))
    
    def forward(self, x):
        x1, x2, x3 = self.darknet(x)

        x = self.conv1(x1)
        output1 = self.output_layer1(x)#1/16

        x = self.upsample1(x)
        x = torch.cat((x2, x), dim=1)
        x = self.conv2(x)
        output2 = self.output_layer2(x)#1/8

        x = self.upsample2(x)
        x = torch.cat((x3, x), dim=1)
        x = self.conv3(x)
        output3 = self.output_layer3(x)#1/4

        """ Noting that output1 is the smallest size
        """
        return output1, output2, output3



class UpSample(nn.Module):
    """ Upsample module
    """

    def __init__(self):
        super(UpSample, self).__init__()
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return x

