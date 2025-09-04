import argparse
import sys
from typing import List

sys.path.append('../')
from backbones.base_backbone import Backbone
from backbones.convnext_unet import ConvNeXtUnet
from backbones.dilated_cnn import DilatedCnn
from backbones.light_hrnet import LightHRNet
from backbones.hrnet import HRNet
from backbones.maxvit_unet import MaxVitPicoUnet, MaxVitUnet
from backbones.mobilenetv3 import MobileNetV3
from backbones.s3k import S3K
from backbones.unet import Unet


class BackboneFactory:
    # TODO: how to auto-register with __init__subclass over multiple files?
    registered_backbone_classes: List[Backbone] = [
        Unet,
        ConvNeXtUnet,
        MaxVitUnet,
        MaxVitPicoUnet,
        S3K,
        DilatedCnn,
        MobileNetV3,
        LightHRNet,
        HRNet,
    ]

    @staticmethod
    def create_backbone(backbone_type: str, confs, **kwargs) -> Backbone:
        for backbone_class in BackboneFactory.registered_backbone_classes:
            if backbone_type == backbone_class.__name__:
                if backbone_type == 'HRNet':
                    return backbone_class(n_channels_in=3, n_channels_out=confs.num_joints, pretrained=confs.hrnet_pretrained, **kwargs)
                else:
                    return backbone_class(**kwargs)
        raise Exception("Unknown backbone type")

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group(BackboneFactory.__name__)
        parser.add_argument("--backbone_type", type=str, required=False, default=Unet.__name__,
                            help="The Class of the Backbone for the Detector." )

        # add all backbone hyperparams.
        for backbone_class in BackboneFactory.registered_backbone_classes:
            parent_parser = backbone_class.add_to_argparse(parent_parser)
        return parent_parser
