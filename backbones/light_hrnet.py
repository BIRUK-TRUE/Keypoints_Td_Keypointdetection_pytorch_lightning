import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.base_backbone import Backbone


class BasicBlock(nn.Module):
    """Basic residual block for HRNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LightHRNet(Backbone):
    """Light HRNet backbone for keypoint detection with proper cross-scale feature exchange"""

    def __init__(self, n_channels_in=3, n_channels=32, num_stages=2, num_branches=2,
                 num_blocks=2, num_channels=None, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.num_stages = num_stages
        self.num_branches = num_branches
        self.num_blocks = num_blocks

        # Set progressive channel configuration if not provided
        if num_channels is None:
            # Standard HRNet progressive channel scaling: [C, 2C, 4C, ...]
            self.num_channels = [n_channels * (2**i) for i in range(num_branches)]
        else:
            self.num_channels = num_channels

        # Initial convolution
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)

        # First stage - single branch
        self.layer1 = self._make_layer(BasicBlock, n_channels, n_channels, num_blocks)

        # Multi-scale stages with proper feature exchange
        self.transition1 = self._make_transition_layer([n_channels], self.num_channels)
        self.stage2 = self._make_stage(self.num_channels, num_blocks)
        self.fuse_layers2 = self._make_fuse_layers(num_branches, self.num_channels)

        if num_stages > 2:
            # For 3+ stages, maintain the same channel structure
            self.transition2 = self._make_transition_layer(self.num_channels, self.num_channels)
            self.stage3 = self._make_stage(self.num_channels, num_blocks)
            self.fuse_layers3 = self._make_fuse_layers(num_branches, self.num_channels)

        # Final fusion layer - properly fuse all scales
        self.final_layer = self._make_final_layer()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Create transition layers between stages with different numbers of branches"""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                # Existing branch - check if channel change is needed
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                # New branch - create from the last existing branch
                inchannels = num_channels_pre_layer[-1]
                transition_layers.append(nn.Sequential(
                    nn.Conv2d(inchannels, num_channels_cur_layer[i], 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_channels_cur_layer[i]),
                    nn.ReLU(inplace=True)))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_channels, num_blocks):
        """Create a stage with multiple branches"""
        num_branches = len(num_channels)
        blocks = []
        for i in range(num_branches):
            blocks.append(self._make_layer(BasicBlock, num_channels[i], num_channels[i], num_blocks))
        return nn.ModuleList(blocks)

    def _make_fuse_layers(self, num_branches, num_channels):
        """Create fusion layers for cross-scale feature exchange"""
        if num_branches == 1:
            return None

        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # Downsample from higher resolution to lower resolution
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # Upsample from lower resolution to higher resolution
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_channels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)
                            ))
                        else:
                            num_outchannels_conv3x3 = num_channels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def _make_final_layer(self):
        """Create final fusion layer that properly combines all scales"""
        # Upsample all branches to highest resolution and sum
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.num_channels[i], self.n_channels, 1, bias=False),
                nn.BatchNorm2d(self.n_channels),
                nn.ReLU(inplace=True)
            ) for i in range(self.num_branches)
        ])

    def forward_stage(self, stage, x_list):
        """Forward pass for a stage (ModuleList of branches) without fusion"""
        y_list = []
        for i, branch in enumerate(stage):
            y_list.append(branch(x_list[i]))
        return y_list

    def forward_stage_with_fusion(self, stage, fuse_layers, x_list):
        """Forward pass for a stage with cross-scale feature exchange"""
        # First process each branch independently
        y_list = []
        for i, branch in enumerate(stage):
            y_list.append(branch(x_list[i]))
        
        # Then fuse features across scales if fusion layers exist
        if fuse_layers is None:
            return y_list
        
        x_fuse = []
        for i in range(len(y_list)):
            y = y_list[i]
            for j in range(len(y_list)):
                if i != j and fuse_layers[i][j] is not None:
                    y = y + fuse_layers[i][j](y_list[j])
            x_fuse.append(self.relu(y))
        
        return x_fuse

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # First stage
        x = self.layer1(x)

        # Transition to multi-scale
        x_list = []
        for i in range(self.num_branches):
            if i < len(self.transition1) and self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # Second stage with feature fusion
        y_list = self.forward_stage_with_fusion(self.stage2, self.fuse_layers2, x_list)

        # If more than 2 stages
        if self.num_stages > 2:
            x_list = []
            for i in range(len(self.transition2)):
                if self.transition2[i] is not None:
                    if i < len(y_list):
                        x_list.append(self.transition2[i](y_list[i]))
                    else:
                        x_list.append(self.transition2[i](y_list[-1]))
                else:
                    if i < len(y_list):
                        x_list.append(y_list[i])
                    else:
                        x_list.append(y_list[-1])
            y_list = self.forward_stage_with_fusion(self.stage3, self.fuse_layers3, x_list)

        # Final fusion: upsample all branches to highest resolution and sum
        x_final = None
        for i in range(len(y_list)):
            # Process each branch through final layer
            branch_out = self.final_layer[i](y_list[i])
            
            # Upsample to match the highest resolution (first branch)
            if i > 0:
                branch_out = F.interpolate(
                    branch_out, 
                    size=y_list[0].shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Sum all branches
            if x_final is None:
                x_final = branch_out
            else:
                x_final = x_final + branch_out

        return x_final

    def get_n_channels_out(self):
        return self.n_channels

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("LightHRNet")
        parser.add_argument("--n_channels", type=int, default=32, help="Number of channels in the backbone")
        parser.add_argument("--num_stages", type=int, default=2, help="Number of stages in HRNet")
        parser.add_argument("--num_branches", type=int, default=2, help="Number of branches in HRNet")
        parser.add_argument("--num_blocks", type=int, default=2, help="Number of blocks per stage")
        return parent_parser
