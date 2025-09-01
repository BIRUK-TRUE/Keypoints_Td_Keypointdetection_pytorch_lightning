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
    """Light HRNet backbone for keypoint detection"""
    
    def __init__(self, n_channels_in=3, n_channels=32, num_stages=2, num_branches=2, 
                 num_blocks=2, num_channels=None, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.num_stages = num_stages
        self.num_branches = num_branches
        self.num_blocks = num_blocks
        
        # Set default channel configuration if not provided
        if num_channels is None:
            self.num_channels = [n_channels] * num_branches
        else:
            self.num_channels = num_channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # First stage - single branch
        self.layer1 = self._make_layer(BasicBlock, n_channels, n_channels, num_blocks)
        
        # Multi-scale stages - create transition from single branch to multi-branch
        self.transition1 = self._make_transition_layer([n_channels], self.num_channels)
        self.stage2 = self._make_stage(self.num_channels, num_blocks)
        
        if num_stages > 2:
            # For 3+ stages, double the channels
            next_channels = [c * 2 for c in self.num_channels]
            self.transition2 = self._make_transition_layer(self.num_channels, next_channels)
            self.stage3 = self._make_stage(next_channels, num_blocks)
        
        # Final fusion layer - sum all branch channels
        total_channels = sum(self.num_channels)
        self.final_layer = nn.Conv2d(total_channels, n_channels, kernel_size=1)
        
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
        
        # Second stage
        y_list = self.stage2(x_list)
        
        # Fuse multi-scale features
        x_fuse = []
        for i in range(len(y_list)):
            if i == 0:
                x_fuse.append(y_list[i])
            else:
                # Upsample to match the first branch
                y = F.interpolate(y_list[i], size=y_list[0].shape[2:], mode='bilinear', align_corners=False)
                x_fuse.append(y)
        
        # Concatenate and fuse
        x = torch.cat(x_fuse, dim=1)
        x = self.final_layer(x)
        
        return x
    
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
