import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class DPTDecoder(nn.Module):
    def __init__(
        self,
        in_channels=[96, 192, 384, 768],  # depends on the backbone, defult for Swin
        fusion_hidden_size=256,
        num_classes=1  # Change based on your segmentation classes
    ):
        
        '''
        This layer takes hidden feature from several hidden layers, and do the progressive fusion to 
        get the final segmentation map/dense prediction.

        The number of hidden layer to be used is determined by the the length of in_channels.

        args:
            in_channels: list of int, the number of channels for each hidden layer.
            fusion_hidden_size: int, the number of channels for the hidden layers in the fusion block.
            num_classes: int, the number of classes for the segmentation task.
        '''
        super().__init__()

        self.in_channels = in_channels
        self.fusion_hidden_size = fusion_hidden_size
        self.num_classes = num_classes
        
        # Readout projections for each stage
        self.readout = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chan, fusion_hidden_size, 1),
                nn.GELU()
            ) for chan in in_channels
        ])
        
        # Fusion blocks
        self.fusion_1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fusion_hidden_size, fusion_hidden_size, 3, padding=1),
                nn.BatchNorm2d(fusion_hidden_size),
                nn.ReLU(inplace=True)
            ) for chan in range(len(in_channels))
        ])
        self.fusion_2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fusion_hidden_size, fusion_hidden_size, 3, padding=1),
                nn.BatchNorm2d(fusion_hidden_size),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels) - 1)
        ])
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(fusion_hidden_size, fusion_hidden_size // 2, 3, padding=1),
            nn.BatchNorm2d(fusion_hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_hidden_size // 2, num_classes, 1)
        )
        
        
    
    def forward(self, features, target_res):
        B = features[0].shape[0]
        
        # Process each feature level
        outs = []
        for i in range(len(features)):
            # Reshape and project features
            feat = features[i]
            if len(feat.shape) == 3:
                
                H = W = int(math.sqrt(feat.shape[1]))
                feat = feat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
 
            # Apply readout and resize
            feat = self.readout[i](feat)
            feat = F.interpolate(feat, size=(target_res[0]//2, target_res[1]//2), mode='bilinear', align_corners=False)
            outs.append(feat)
        
        # Progressive fusion
        fused = outs[-1]
        for i in range(len(outs) - 2, 0, -1):
            current = outs[i]
            current = self.fusion_1[i](current) + current

            fused = current + fused
            fused = self.fusion_2[i](fused) + fused

        # Final projection'
        fused = nn.functional.interpolate(fused, size=(target_res[0], target_res[1]), mode='bilinear', align_corners=False)
        return self.head(fused)