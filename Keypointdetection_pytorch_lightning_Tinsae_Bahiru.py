""" avoid circular imports by separating types"""
import abc
import albumentations as alb
import argparse
from collections import defaultdict
import json
import math
import numpy as np
import os
from pathlib import Path
from pydantic import BaseModel
import pytorch_lightning as pl
import random
from skimage import io
import time
import timm
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor
import typing
from typing import List, Optional, Tuple, Union


KEYPOINT_TYPE = Tuple[int, int]  # (u,v)
COCO_KEYPOINT_TYPE = Tuple[int, int, int]  # (u,v,f)
CHANNEL_KEYPOINTS_TYPE = List[KEYPOINT_TYPE]
IMG_KEYPOINTS_TYPE = List[CHANNEL_KEYPOINTS_TYPE]

"""Custom parser for COCO keypoints JSON"""
LicenseID = int
ImageID = int
CategoryID = int
AnnotationID = int
Segmentation = List[List[Union[float, int]]]
FileName = str
Relativepath = str
Url = str


class MultiChannelKeypointsCompose(alb.Compose):
    """A subclass of Albumentations.Compose to accomodate for multiple groups/channels of keypoints.
    Some transforms (crop e.g.) will result in certain keypoints no longer being in the new image. Albumentations can remove them, but since it operates
    on a single list of keypoints, the transformed keypoints need to be associated to their channel afterwards. Albumentations has support for labels to accomodate this,
    so we label each keypoint with the index of its channel.
    """

    def __init__(self, transforms, p: float = 1):
        keypoint_params = alb.KeypointParams(format="xy", label_fields=["channel_labels"], remove_invisible=True)
        super().__init__(transforms, keypoint_params=keypoint_params, p=p)

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:

        # flatten and create channel labels (=str(index))
        keypoints = data["keypoints"]
        self.create_channel_labels(keypoints)
        flattened_keypoints = self.flatten_keypoints(keypoints)
        data["keypoints"] = flattened_keypoints
        data["channel_labels"] = self.flatten_keypoints(self.create_channel_labels(keypoints))
        # apply transforms
        result_dict = super().__call__(*args, force_apply=force_apply, **data)

        # rearrange keypoints by channel
        transformed_flattened_keypoints = result_dict["keypoints"]
        transformed_flattened_labels = result_dict["channel_labels"]
        transformed_keypoints = self.order_transformed_keypoints_by_channel(
            keypoints, transformed_flattened_keypoints, transformed_flattened_labels
        )
        result_dict["keypoints"] = transformed_keypoints
        return result_dict

    @staticmethod
    def flatten_keypoints(keypoints: IMG_KEYPOINTS_TYPE) -> List[KEYPOINT_TYPE]:
        return [item for sublist in keypoints for item in sublist]

    @staticmethod
    def create_channel_labels(keypoints: IMG_KEYPOINTS_TYPE):
        channel_labels = [[str(i)] * len(keypoints[i]) for i in range(len(keypoints))]
        return channel_labels

    @staticmethod
    def order_transformed_keypoints_by_channel(
        original_keypoints: IMG_KEYPOINTS_TYPE,
        transformed_keypoints: List[KEYPOINT_TYPE],
        transformed_channel_labels: List[str],
    ) -> IMG_KEYPOINTS_TYPE:
        ordered_transformed_keypoints = [[] for _ in original_keypoints]
        for transformed_keypoint, channel_label in zip(transformed_keypoints, transformed_channel_labels):
            channel_idx = int(channel_label)
            ordered_transformed_keypoints[channel_idx].append(transformed_keypoint)

        return ordered_transformed_keypoints


class CocoInfo(BaseModel):
    description: str
    url: Url
    version: str
    year: int
    contributor: str
    date_created: str


class CocoLicenses(BaseModel):
    url: Url
    id: LicenseID
    name: str


class CocoImage(BaseModel):
    license: Optional[LicenseID] = None
    file_name: Relativepath
    height: int
    width: int
    id: ImageID


class CocoKeypointCategory(BaseModel):
    supercategory: str  # should be set to "name" for root category
    id: CategoryID
    name: str
    keypoints: List[str]
    skeleton: Optional[List[List[int]]] = None


class CocoKeypointAnnotation(BaseModel):
    category_id: CategoryID
    id: AnnotationID
    image_id: ImageID

    num_keypoints: Optional[int] = None
    # COCO keypoints can be floats if they specify the exact location of the keypoint (e.g. from CVAT)
    # even though COCO format specifies zero-indexed integers
    # (i.e. every keypoint in the [0,1]x [0.1] pixel box becomes (0,0)
    keypoints: List[float]

    # TODO: add checks.
    # @validator("keypoints")
    # def check_amount_of_keypoints(cls, v, values, **kwargs):
    #     assert len(v) // 3 == values["num_keypoints"]


class CocoKeypoints(BaseModel):
    """Parser Class for COCO keypoints JSON

    Example:
    with open("path","r") as file:
        data = json.load(file) # dict
        parsed_data = COCOKeypoints(**data)
    """

    info: Optional[CocoInfo] = None
    licenses: Optional[List[CocoLicenses]] = None
    images: List[CocoImage]
    categories: List[CocoKeypointCategory]
    annotations: List[CocoKeypointAnnotation]


class ImageLoader:
    def get_image(self, path: str, idx: int) -> np.ndarray:
        """
        read the image from disk and return as np array
        """
        # load images @runtime from disk
        image = io.imread(path)
        return image


class BaseImageLoaderDecorator(ImageLoader):
    def __init__(self, image_loader: ImageLoader) -> None:
        self.image_loader = image_loader

    @abc.abstractmethod
    def get_image(self, path: str, idx: int) -> np.ndarray:
        pass


class IOSafeImageLoaderDecorator(BaseImageLoaderDecorator):
    """
    IO safe loader that re-attempts to load image from disk (important for GPULab infrastructure @ UGent)
    """

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__(image_loader)
        self.n_io_attempts = 4

    def get_image(self, path: str, idx: int) -> np.ndarray:
        sleep_time_in_seconds = 1
        for j in range(self.n_io_attempts):
            try:
                image = self.image_loader.get_image(path, idx)
                return image
            except IOError:
                if j == self.n_io_attempts - 1:
                    raise IOError(f"Could not load image for dataset entry with path {path}, index {idx}")

                sleep_time = max(random.gauss(sleep_time_in_seconds, j), 0)
                print(f"caught IOError in {j}th attempt to load image for {path}, sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
                sleep_time_in_seconds *= 2


class CachedImageLoaderDecorator(BaseImageLoaderDecorator):
    """
    Image dataloader that caches the images after the first fetch in np.uint8 format.
    Requires enough CPU Memory to fit entire dataset (img_size^2*3*N_images B)

    This is done lazy instead of prefetching, as the torch dataloader is highly optimized to prefetch data during forward passes etc.
     Impact is expected to be not too big.. TODO -> benchmark.

    Furthermore, this caching requires to set num_workers to 0, as the dataset object is copied by each dataloader worker.
    """

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__(image_loader)

        self.cache = []
        self.cache_index_mapping = {}

    def get_image(self, path: str, idx: int) -> np.ndarray:
        if path not in self.cache_index_mapping:
            img = super().get_image(path, idx)
            self.cache.append(img)
            self.cache_index_mapping.update({path: len(self.cache) - 1})
            return img

        else:
            return self.cache[self.cache_index_mapping[path]]


class ImageDataset(Dataset, abc.ABC):
    def __init__(self, imageloader: ImageLoader = None):
        if imageloader is None:
            self.image_loader = IOSafeImageLoaderDecorator(ImageLoader())

        else:
            assert isinstance(imageloader, ImageLoader)
            self.image_loader = imageloader

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class UnlabeledKeypointsDataset(ImageDataset):
    """
    Simple dataset to run inference on unlabeled data
    """

    def __init__(
        self,
        image_dataset_path: str,
        **kwargs,
    ):
        super().__init__()
        self.image_paths = os.listdir(image_dataset_path)
        self.image_paths = [image_dataset_path + f"/{path}" for path in self.image_paths]

        self.transform = ToTensor()  # convert images to Torch Tensors

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image_path = self.image_paths[index]
        image = self.image_loader.get_image(image_path, index)
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)


class KeypointsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        """
        parser = parent_parser.add_argument_group("KeypointsDatamodule")
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--validation_split_ratio", default=0.25, type=float)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument(
            "--json_dataset_path",
            type=str,
            help="Absolute path to the json file that defines the train dataset according to the COCO format.",
            required=True,
        )
        parser.add_argument(
            "--json_validation_dataset_path",
            type=str,
            help="Absolute path to the json file that defines the validation dataset according to the COCO format. \
                If not specified, the train dataset will be split to create a validation set if there is one.",
        )
        parser.add_argument(
            "--json_test_dataset_path",
            type=str,
            help="Absolute path to the json file that defines the test dataset according to the COCO format. \
                If not specified, no test set evaluation will be performed at the end of training.",
        )

        parser.add_argument("--augment_train", dest="augment_train", default=False, action="store_true")
        parent_parser = COCOKeypointsDataset.add_argparse_args(parent_parser)

        return parent_parser

    def __init__(
        self,
        keypoint_channel_configuration: list[list[str]],
        batch_size: int = 16,
        validation_split_ratio: float = 0.25,
        num_workers: int = 2,
        json_dataset_path: str = None,
        json_validation_dataset_path: str = None,
        json_test_dataset_path=None,
        augment_train: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train

        self.train_dataset = None
        if json_dataset_path:
            self.train_dataset = COCOKeypointsDataset(json_dataset_path, keypoint_channel_configuration, **kwargs)

        self.validation_dataset = None
        self.test_dataset = None

        if json_validation_dataset_path:
            self.validation_dataset = COCOKeypointsDataset(
                json_validation_dataset_path, keypoint_channel_configuration, **kwargs
            )
        else:
            if self.train_dataset is not None:
                print(f"splitting the train set to create a validation set with ratio {validation_split_ratio} ")
                self.train_dataset, self.validation_dataset = KeypointsDataModule._split_dataset(
                    self.train_dataset, validation_split_ratio
                )

        if json_test_dataset_path:
            self.test_dataset = COCOKeypointsDataset(json_test_dataset_path, keypoint_channel_configuration, **kwargs)

        # create the transforms if needed and set them to the datasets
        if augment_train:
            print("Augmenting the training dataset!")
            img_height, img_width = self.train_dataset[0][0].shape[1], self.train_dataset[0][0].shape[2]
            aspect_ratio = img_width / img_height
            train_transform = MultiChannelKeypointsCompose(
                [
                    alb.ColorJitter(p=0.8),
                    alb.RandomBrightnessContrast(p=0.8),
                    alb.RandomResizedCrop(
                        img_height, img_width, scale=(0.8, 1.0), ratio=(0.9 * aspect_ratio, 1.1 * aspect_ratio), p=1.0
                    ),
                    alb.GaussianBlur(p=0.2, blur_limit=(3, 3)),
                    alb.Sharpen(p=0.2),
                    alb.GaussNoise(),
                ]
            )
            if isinstance(self.train_dataset, COCOKeypointsDataset):
                self.train_dataset.transform = train_transform
            elif isinstance(self.train_dataset, Subset):
                # if the train dataset is a subset, we need to set the transform to the underlying dataset
                # otherwise the transform will not be applied..
                assert isinstance(self.train_dataset.dataset, COCOKeypointsDataset)
                self.train_dataset.dataset.transform = train_transform

    @staticmethod
    def _split_dataset(dataset, validation_split_ratio):
        validation_size = int(validation_split_ratio * len(dataset))
        train_size = len(dataset) - validation_size
        train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
        print(f"train size: {len(train_dataset)}")
        print(f"validation size: {len(validation_dataset)}")
        return train_dataset, validation_dataset

    def train_dataloader(self):
        # usually need to seed workers for reproducibility
        # cf. https://pytorch.org/docs/stable/notes/randomness.html
        # but PL does for us in their seeding function:
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility

        if self.train_dataset is None:
            return None

        dataloader = DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=COCOKeypointsDataset.collate_fn,
            pin_memory=True,  # usually a little faster
        )
        return dataloader

    def val_dataloader(self):
        # usually need to seed workers for reproducibility
        # cf. https://pytorch.org/docs/stable/notes/randomness.html
        # but PL does for us in their seeding function:
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility

        if self.validation_dataset is None:
            return None

        dataloader = DataLoader(
            self.validation_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=COCOKeypointsDataset.collate_fn,
        )
        return dataloader

    def test_dataloader(self):

        if self.test_dataset is None:
            return None
        dataloader = DataLoader(
            self.test_dataset,
            min(4, self.batch_size),  # 4 as max for better visualization in wandb.
            shuffle=False,
            num_workers=0,
            collate_fn=COCOKeypointsDataset.collate_fn,
        )
        return dataloader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class COCOKeypointsDataset(ImageDataset):
    """Pytorch Dataset for COCO-formatted Keypoint dataset

    cf. https://cocodataset.org/#format-data for more information. We expect each annotation to have at least the keypoints and num_keypoints fields.
    Each category should also have keypoints. For more information on the required fields and data types, have a look at the COCO parser in `coco_parser.py`.

    The dataset builds an index during the init call that maps from each image_id to a list of all keypoints of all semantic types in the dataset.

    The Dataset also expects a keypoint_channel_configuration that maps from the semantic types (the keypoints in all categories of the COCO file) to the channels
    of the keypoint detector. In the simplest case this is simply a list of all types, but for e.g. symmetric objects or equivalence mapping one could combine different
    types into one channel. For example if you have category box with keypoints [corner0, corner1, corner2, corner3] you could combine  them in a single channel for the
    detector by passing as configuration [[corner0,corner1,corner2,corner3]].

    You can also select if you want to train on annotations with flag=1 (occluded).

    The paths in the JSON should be relative to the directory in which the JSON is located.


    The __getitem__ function returns [img_path, [keypoints for each channel according to the configuration]]
    """

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        add named arguments from the init function to the parser
        """
        parser = parent_parser.add_argument_group("COCOkeypointsDataset")
        parser.add_argument(
            "--detect_only_visible_keypoints",
            dest="detect_only_visible_keypoints",
            default=False,
            action="store_true",
            help="If set, only keypoints with flag > 1.0 will be used.",
        )

        return parent_parser

    def __init__(
        self,
        json_dataset_path: str,
        keypoint_channel_configuration: list[list[str]],
        detect_only_visible_keypoints: bool = True,
        transform: alb.Compose = None,
        imageloader: ImageLoader = None,
        **kwargs,
    ):
        super().__init__(imageloader)

        self.image_to_tensor_transform = ToTensor()
        self.dataset_json_path = Path(json_dataset_path)
        self.dataset_dir_path = self.dataset_json_path.parent  # assume paths in JSON are relative to this directory!

        self.keypoint_channel_configuration = keypoint_channel_configuration
        self.detect_only_visible_keypoints = detect_only_visible_keypoints

        print(f"{detect_only_visible_keypoints=}")

        self.random_crop_transform = None
        self.transform = transform
        self.dataset = self.prepare_dataset()  # idx: (image, list(keypoints/channel))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, IMG_KEYPOINTS_TYPE]:
        """
        Returns:
            (image, keypoints); image = 3xHxW tensor; keypoints = List(c x list( list of K_i keypoints ))

            e.g. for 2 heatmap channels with respectively 1,2 keypoints, the keypoints list will be formatted as
            [[[u11,v11]],[[u21,v21],[u22,v22]]]
        """
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image_path = self.dataset_dir_path / self.dataset[index][0]
        image = self.image_loader.get_image(str(image_path), index)
        # remove a-channel if needed
        if image.shape[2] == 4:
            image = image[..., :3]

        keypoints = self.dataset[index][1]

        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image, keypoints = transformed["image"], transformed["keypoints"]

        # convert all keypoints to integers values.
        # COCO keypoints can be floats if they specify the exact location of the keypoint (e.g. from CVAT)
        # even though COCO format specifies zero-indexed integers (i.e. every keypoint in the [0,1]x [0.1] pixel box becomes (0,0)
        # we convert them to ints here, as the heatmap generation will add a 0.5 offset to the keypoint location to center it in the pixel
        # the distance metrics also operate on integer values.

        # so basically from here on every keypoint is an int that represents the pixel-box in which the keypoint is located.
        keypoints = [
            [[math.floor(keypoint[0]), math.floor(keypoint[1])] for keypoint in channel_keypoints]
            for channel_keypoints in keypoints
        ]
        image = self.image_to_tensor_transform(image)
        return image, keypoints

    def prepare_dataset(self):
        """Prepares the dataset to map from COCO to (img, [keypoints for each channel])

        Returns:
            [img_path, [list of keypoints for each channel]]
        """
        with open(self.dataset_json_path, "r") as file:
            data = json.load(file)
            parsed_coco = CocoKeypoints(**data)

            img_dict: typing.Dict[int, CocoImage] = {}
            for img in parsed_coco.images:
                img_dict[img.id] = img

            category_dict: typing.Dict[int, CocoKeypointCategory] = {}
            for category in parsed_coco.categories:
                category_dict[category.id] = category

            # iterate over all annotations and create a dict {img_id: {semantic_type : [keypoints]}}
            # make sure to deal with multiple occurances of same semantic_type in one image (e.g. multipe humans in one image)
            annotation_dict = defaultdict(
                lambda: defaultdict(lambda: [])
            )  # {img_id: {channel : [keypoints for that channel]}}
            for annotation in parsed_coco.annotations:
                # add all keypoints from this annotation to the corresponding image in the dict

                img = img_dict[annotation.image_id]
                category = category_dict[annotation.category_id]
                semantic_classes = category.keypoints

                keypoints = annotation.keypoints
                keypoints = self.split_list_in_keypoints(keypoints)
                for semantic_type, keypoint in zip(semantic_classes, keypoints):
                    annotation_dict[annotation.image_id][semantic_type].append(keypoint)

            # iterate over each image and all it's annotations
            # filter the visible keypoints
            # and group them by channel
            dataset = []
            for img_id, keypoint_dict in annotation_dict.items():
                img_channels_keypoints = [[] for _ in range(len(self.keypoint_channel_configuration))]
                for semantic_type, keypoints in keypoint_dict.items():
                    for keypoint in keypoints:

                        if min(keypoint[:2]) < 0 or keypoint[0] > img_dict[img_id].width or keypoint[1] > img_dict[img_id].height:
                            print("keypoint outside of image, ignoring.")
                            continue
                        if self.is_keypoint_visible(keypoint):
                            channel_idx = self.get_keypoint_channel_index(semantic_type)
                            if channel_idx > -1:
                                img_channels_keypoints[channel_idx].append(keypoint[:2])

                dataset.append([img_dict[img_id].file_name, img_channels_keypoints])

            return dataset

    def get_keypoint_channel_index(self, semantic_type: str) -> int:
        """
        given a semantic type, get it's channel according to the channel configuration.
        Returns -1 if the semantic type couldn't be found.
        """

        for i, types_in_channel in enumerate(self.keypoint_channel_configuration):
            if semantic_type in types_in_channel:
                return i
        return -1

    def is_keypoint_visible(self, keypoint: COCO_KEYPOINT_TYPE) -> bool:
        """
        Args:
            keypoint (list): [u,v,flag]

        Returns:
            bool: True if current keypoint is considered visible according to the dataset configuration, else False
        """
        if self.detect_only_visible_keypoints:
            # filter out occluded keypoints with flag 1.0
            return keypoint[2] > 1.5
        else:
            # filter out non-labeled keypoints with flag 0.0
            return keypoint[2] > 0.5

    @staticmethod
    def split_list_in_keypoints(list_to_split: List[COCO_KEYPOINT_TYPE]) -> List[List[COCO_KEYPOINT_TYPE]]:
        """
        splits list [u1,v1,f1,u2,v2,f2,...] to [[u,v,f],..]
        """
        n = 3
        output = [list_to_split[i : i + n] for i in range(0, len(list_to_split), n)]
        return output

    @staticmethod
    def collate_fn(data):
        """custom collate function for use with the torch dataloader

        Note that it could have been more efficient to padd for each channel separately, but it's not worth the trouble as even
        for 100 channels with each 100 occurances the padded data size is still < 1kB..

        Args:
            data: list of tuples (image, keypoints); image = 3xHxW tensor; keypoints = List(c x list(? keypoints ))

        Returns:
            (images, keypoints); Images as a torch tensor Nx3xHxW,
            keypoints is a nested list of lists. where each item is a tensor (K,2) with K the number of keypoints
            for that channel and that sample:

                List(List(Tensor(K,2))) -> C x N x Tensor(max_keypoints_for_any_channel_in_batch x 2)

        Note there is no padding, as all values need to be unpacked again in the detector to create all the heatmaps,
        unlike e.g. NLP where you directly feed the padded sequences to the network.
        """
        images, keypoints = zip(*data)

        # convert the list of keypoints to a 2D tensor
        keypoints = [[torch.tensor(x) for x in y] for y in keypoints]
        # reorder to have the different keypoint channels as  first dimension
        # C x N x K x 2 , K = variable number of keypoints for each (N,C)
        reordered_keypoints = [[keypoints[i][j] for i in range(len(keypoints))] for j in range(len(keypoints[0]))]

        images = torch.stack(images)

        return images, reordered_keypoints


class Backbone(nn.Module, abc.ABC):
    """Base class for backbones"""

    def __init__(self):
        super(Backbone, self).__init__()

    @abc.abstractmethod
    def get_n_channels_out(self) -> int:
        raise NotImplementedError

    @staticmethod
    def add_to_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parent_parser


class UpSamplingBlock(nn.Module):
    """A Unet-like backbone that uses a (relatively) small imagenet-pretrained ConvNeXt model from timm as encoder. """
    """
    A very basic Upsampling block (these params have to be learnt from scratch so keep them small)

    First it reduces the number of channels of the incoming layer to the amount of the skip connection with a 1x1 conv
    then it concatenates them and combines them in a new conv layer.



    x --> up ---> conv1 --> concat --> conv2 --> norm -> relu
                  ^
                  |
                  skip_x
    """

    def __init__(self, n_channels_in, n_skip_channels_in, n_channels_out, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=n_skip_channels_in + n_channels_in,
            out_channels=n_channels_out,
            kernel_size=kernel_size,
            bias=False,
            padding="same",
        )

        self.norm1 = nn.BatchNorm2d(n_channels_out)
        self.relu1 = nn.ReLU()

    def forward(self, x, x_skip):
        # bilinear is not deterministic, use nearest neighbor instead
        x = nn.functional.interpolate(x, scale_factor=2.0)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # second conv as in original UNet upsampling block decreases performance
        # probably because I was using a small dataset that did not have enough data to learn the extra parameters
        return x


class ConvNeXtUnet(Backbone):
    """
    Pretrained ConvNeXt as Encoder for the U-Net.

    the outputs of the 3 intermediate CovNext stages are used for skip connections.
    The output of res4 is considered as the bottleneck and has a 32x resolution reduction!

    femto -> 3M params
    nano -> 17M params (but only twice as slow)


    input                                                   final_conv --- head
        stem                                            upsampling
                                                    upsamping
            res1         --->   1/4             decode3
                res2     --->   1/8         decode2
                    res3 --->   1/16    decode1
                        res4 ---1/32----|
    """

    def __init__(self, **kwargs):
        super().__init__()
        # todo: make desired convnext encoder configurable
        self.encoder = timm.create_model("convnext_femto", features_only=True, pretrained=True)

        self.decoder_blocks = nn.ModuleList()
        for i in range(1, 4):
            channels_in, skip_channels_in = (
                self.encoder.feature_info.info[-i]["num_chs"],
                self.encoder.feature_info.info[-i - 1]["num_chs"],
            )
            block = UpSamplingBlock(channels_in, skip_channels_in, skip_channels_in, 3)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Conv2d(skip_channels_in + 3, skip_channels_in, 3, padding="same")

    def forward(self, x):
        x_orig = torch.clone(x)
        features = self.encoder(x)

        x = features.pop()
        for block in self.decoder_blocks:
            x = block(x, features.pop())
        x = nn.functional.interpolate(x, scale_factor=4.0)
        x = torch.cat([x, x_orig], dim=1)
        x = self.final_conv(x)
        return x

    def get_n_channels_out(self):
        return self.encoder.feature_info.info[0]["num_chs"]


class DilatedCnn(Backbone):
    """A very simple Backbone that uses dilated CNNs without spatial resolution changes.
    """
    def __init__(self, n_channels=32, **kwargs):
        super().__init__()
        self.n_channels_in = 3
        self.n_channels = n_channels
        kernel_size = (3, 3)
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.n_channels_in,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=2,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=4,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=8,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=16,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=2,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=4,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=8,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=16,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_n_channels_out(self):
        return self.n_channels





