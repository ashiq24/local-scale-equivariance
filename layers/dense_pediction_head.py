import torch 
import torch.nn as nn
import torch.nn.functional as F

class LinearDensePredHead(nn.Module):
    def __init__(self, 
                inchannels: int,
                outchannels: int,
                ):
        super().__init__()
        '''
        Linear segmentation head, 1x1 conv, upsample to the original image size
        '''
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, padding=0)

    def forward(self, x, output_shape):
        x = self.conv1(x)
        x = F.interpolate(x, size=output_shape, mode='bilinear', align_corners=False)
        return x
