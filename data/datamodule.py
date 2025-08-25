import albumentations as alb
import argparse
import numpy as np
import pytorch_lightning as pl
import random
import sys
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append('../')
import config_file as confs
from data.augmentations import MultiChannelKeypointsCompose
from data.coco_dataset import COCOKeypointsDataset


class KeypointsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        """
        parser = parent_parser.add_argument_group("KeypointsDatamodule")
        parser.add_argument("--batch_size", required=False, default=16, type=int)
        parser.add_argument("--validation_split_ratio", required=False, default=0.25, type=float)
        parser.add_argument("--num_workers", required=False, default=4, type=int)
        parser.add_argument("--json_dataset_path", type=str, required=True,
            help="Absolute path to the json file that defines the train dataset according to the COCO format.", )
        parser.add_argument("--json_validation_dataset_path", type=str, required=False,
            help="Absolute path to the json file that defines the validation dataset according to the COCO format. "
                 "If not specified, the train dataset will be split to create a validation set if there is one.", )
        parser.add_argument("--json_test_dataset_path", type=str, required=False,
            help="Absolute path to the json file that defines the test dataset according to the COCO format. "
                 "If not specified, no test set evaluation will be performed at the end of training.", )

        parser.add_argument("--augment_train", dest="augment_train", required=False, default=False,
                            action="store_true")
        parent_parser = COCOKeypointsDataset.add_argparse_args(parent_parser)

        return parent_parser

    def __init__(self, keypoint_channel_configuration: list[list[str]], json_dataset_path: str = None,
                 json_val_dataset_path: str = None, json_test_dataset_path=None, val_split_ratio: float = 0.25,
                 batch_size: int = 16, num_workers: int = 2, augment_train: bool = True, **kwargs, ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train

        self.train_dataset = None
        if json_dataset_path:
            self.train_dataset = COCOKeypointsDataset(json_dataset_path, keypoint_channel_configuration, **kwargs)

        self.val_dataset = None
        self.test_dataset = None

        if json_val_dataset_path:
            self.val_dataset = COCOKeypointsDataset(json_val_dataset_path, keypoint_channel_configuration, **kwargs )
        else:
            if self.train_dataset is not None:
                print(f"splitting the train set to create a validation set with ratio {val_split_ratio} ")
                self.train_dataset, self.val_dataset = KeypointsDataModule._split_dataset(self.train_dataset,
                                                                                          val_split_ratio )

        if json_test_dataset_path:
            self.test_dataset = COCOKeypointsDataset(json_test_dataset_path, keypoint_channel_configuration, **kwargs)

        # create the transforms if needed and set them to the datasets
        if augment_train:
            print("Augmenting the training dataset!")
            # img_height, img_width = self.train_dataset[0][0].shape[1], self.train_dataset[0][0].shape[2]
            # img_height, img_width = self.train_dataset.dataset[0][1], self.train_dataset.dataset[0][2]
            img_height, img_width = confs.img_height, confs.img_width
            # aspect_ratio = confs.img_max_width / confs.img_max_height
            aspect_ratio = img_width / img_height
            train_transform = MultiChannelKeypointsCompose([alb.ColorJitter(p=0.8),
                                                            alb.RandomBrightnessContrast(p=0.8),
                                                            # alb.RandomResizedCrop(img_height, img_width,
                                                            #                       scale=(0.8, 1.0),
                                                            #                       ratio=(0.9 * aspect_ratio,
                                                            #                              1.1 * aspect_ratio),
                                                            #                       p=1.0 ),
                                                            alb.GaussianBlur(p=0.2, blur_limit=(3, 3)),
                                                            alb.Sharpen(p=0.2),
                                                            alb.GaussNoise(), ] )
            if isinstance(self.train_dataset, COCOKeypointsDataset):
                self.train_dataset.transform = train_transform
            elif isinstance(self.train_dataset, Subset):
                # if the train dataset is a subset, we need to set the transform to the underlying dataset
                # otherwise the transform will not be applied..
                assert isinstance(self.train_dataset.dataset, COCOKeypointsDataset)
                self.train_dataset.dataset.transform = train_transform

    @staticmethod
    def _split_dataset(dataset, val_split_ratio):
        val_size = int(val_split_ratio * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        print(f"train size: {len(train_dataset)}")
        print(f"validation size: {len(val_dataset)}")

        return train_dataset, val_dataset

    def train_val_dataloader(self):
        # usually need to seed workers for reproducibility
        # cf. https://pytorch.org/docs/stable/notes/randomness.html
        # but PL does for us in their seeding function:
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility

        if self.train_dataset is None:
            return None

        train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers,
                                      collate_fn=COCOKeypointsDataset.collate_fn,
                                      pin_memory=True, )  # usually a little faster

        if self.val_dataset is None:
            return None

        val_dataloader = DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers,
                                    collate_fn=COCOKeypointsDataset.collate_fn, )

        return train_dataloader, val_dataloader

    def train_dataloader(self):
        # usually need to seed workers for reproducibility
        # cf. https://pytorch.org/docs/stable/notes/randomness.html
        # but PL does for us in their seeding function:
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility

        if self.train_dataset is None:
            return None

        dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers,
                                collate_fn=COCOKeypointsDataset.collate_fn,
                                pin_memory=True, )  # usually a little faster

        return dataloader

    def val_dataloader(self):
        # usually need to seed workers for reproducibility
        # cf. https://pytorch.org/docs/stable/notes/randomness.html
        # but PL does for us in their seeding function:
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility

        if self.val_dataset is None:
            return None

        dataloader = DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers,
                                collate_fn=COCOKeypointsDataset.collate_fn, )

        return dataloader

    def test_dataloader(self):

        if self.test_dataset is None:
            return None
        dataloader = DataLoader(self.test_dataset, min(4, self.batch_size), # 4 as max for better visualization in wandb
                                shuffle=False, num_workers=0,
                                collate_fn=COCOKeypointsDataset.collate_fn, )

        return dataloader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
