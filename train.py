# https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da
import albumentations as alb  # Library for augmentations
import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models as pre_models
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as torch_func
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import tqdm
# from tqdm.auto import tqdm

from backbones.backbone_factory import BackboneFactory
import config_file as confs
from data import datamodule, coco_dataset, coco_parser
from models.detector import KeypointDetector
from train_utils import RelativeEarlyStopping, ModelCheckpoint
from utils_file import Averager, SaveBestModel, save_model, save_loss_plot, save_mAP, load_checkpoint

plt.ion()   # This is the interactive mode

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=confs.epochs, required=False,
                    help='Number of epochs to train our network for')
parser.add_argument('-lr', '--learning_rate', type=float, dest='learning_rate', required=False,
                    default=confs.init_lr, help='Learning rate for training the model')
parser.add_argument("--seed", default=2022, required=False, help="seed for reproducibility")
parser.add_argument("--wandb_project", default="keypoint-detection", required=False,
                    help="The wandb project to log the results to" )
parser.add_argument("--wandb_entity", default=None, required=False,
                    help="The entity name to log the project against, can be simply set to your username if you have no dedicated entity for this project", )
parser.add_argument("--wandb_name", default=None, required=False,
                    help="The name of the run, if not specified, a random name will be generated" )
parser.add_argument("--keypoint_channel_configuration", type=str, required=False,
                    help="A list of the semantic keypoints that you want to learn in each channel. These semantic categories must be defined in the COCO dataset. Seperate the channels with a : and the categories within a channel with a =", )

parser.add_argument("--early_stopping_relative_threshold", type=float, required=False,
                    default=-1.0,  # no early stopping by default
                    help="relative threshold for early stopping callback. If validation epoch loss does not increase with at least this fraction compared to the best result so far for 5 consecutive epochs, training is stopped.", )

parser.add_argument("--non-deterministic-pytorch", action="store_false", dest="deterministic", required=False,
                    help="do not use deterministic algorithms for pytorch. This can speed up training, but will make it non-reproducible.",)

parser.add_argument("--wandb_checkpoint_artifact", type=str, required=False,
                    help="A checkpoint to resume/start training from. keep in mind that you currently cannot specify hyperparameters other than the LR.", )
parser.add_argument("--resume_checkpoint", type=str, required=False,
                    help="Path to checkpoint file to resume training from. Should be a .pth file saved by SaveBestModel.", )
args = vars(parser.parse_args())

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)


# Training function.
def train(model):
    print('Training')
    # set the models in training mode
    model.train()

    # initialize tqdm progress bar
    prog_bar = tqdm.tqdm(train_loader, total=len(train_loader))
    for idx, batch in enumerate(prog_bar):
        # batch is (images, keypoints) from COCOKeypointsDataset.collate_fn
        imgs, keypoints = batch

        # compute loss via model's shared_step (handles device internally)
        result = model.shared_step((imgs, keypoints), idx, include_visualization_data_in_result_dict=False)
        loss = result["loss"]
        loss_value = float(loss.detach().cpu())

        train_loss_hist.send(loss_value)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return loss_value


# Validation function.
def validate(model):
    print('\n Validation')
    # set the models in validation mode
    model.eval()

    prog_bar = tqdm.tqdm(val_loader, total=len(val_loader))
    val_losses = []
    # reset validation AP metrics before accumulation
    for m in model.ap_validation_metrics:
        m.reset()

    with torch.no_grad():
        for jdx, batch in enumerate(prog_bar):
            imgs, keypoints = batch
            # include visualization tensors so we can extract predicted heatmaps and gt keypoints for AP
            result = model.shared_step((imgs, keypoints), jdx, include_visualization_data_in_result_dict=True)
            loss = result["loss"].detach().cpu().item()
            val_losses.append(loss)

            # accumulate AP metrics over the validation set
            model.update_ap_metrics(result, model.ap_validation_metrics)

    # compute mean APs across channels and thresholds
    # each metric.compute() returns a dict {threshold: ap}
    per_channel_maps = []
    per_channel_map50 = []
    threshold_keys = None
    for ch_metric in model.ap_validation_metrics:
        ch_ap_dict = ch_metric.compute()
        if threshold_keys is None:
            threshold_keys = list(ch_ap_dict.keys())
        # mean over thresholds for this channel
        ch_mean_ap = float(sum(ch_ap_dict.values()) / max(1, len(ch_ap_dict))) if len(ch_ap_dict) > 0 else 0.0
        per_channel_maps.append(ch_mean_ap)
        # take the first threshold as the "50" proxy (matches first in maximal_gt_keypoint_pixel_distances)
        if len(ch_ap_dict) > 0:
            first_thr_key = threshold_keys[0]
            per_channel_map50.append(float(ch_ap_dict[first_thr_key]))
        else:
            per_channel_map50.append(0.0)
        ch_metric.reset()

    mean_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0
    # aggregate over channels
    map_mean = float(np.mean(per_channel_maps)) if len(per_channel_maps) > 0 else 0.0
    map_50_mean = float(np.mean(per_channel_map50)) if len(per_channel_map50) > 0 else 0.0
    return {"map": map_mean, "map_50": map_50_mean, "val_loss": mean_loss}


if __name__ == '__main__':
    train_json_path = confs.joints_def
    val_json_path = confs.ann_path + confs.dataset_phase[1] + '.json'

    # train_datasets = coco_dataset.COCOKeypointsDataset(train_json_path, confs.all_joints)
    all_dataset = datamodule.KeypointsDataModule(confs.joints_name, train_json_path)
    print(f"[INFO] found {len(all_dataset.train_dataset.dataset)} images in the training set...")
    print(f"[INFO] found {len(all_dataset.val_dataset.dataset)} images in the validation set...")
    # [INFO] found 4 images in the training set...
    # [INFO] found 4 images in the validation set...
    # THERE IS ERROR CHECK THIS  DON'T FORGET
    train_loader, val_loader = all_dataset.train_val_dataloader()

    # Learning_parameters. lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {args['learning_rate']}")
    print(f"Epochs to train for: {epochs}\n")

    # Create backbone using the factory
    backbone_params = {}
    if confs.model_type == 'LightHRNet':
        backbone_params = {
            'n_channels': confs.light_hrnet_channels,
            'num_stages': confs.light_hrnet_stages,
            'num_branches': confs.light_hrnet_branches,
            'num_blocks': confs.light_hrnet_blocks
        }

    backbone_model = BackboneFactory.create_backbone(confs.model_type, confs, **backbone_params)
    print(f"Using {confs.model_type} backbone.")
    my_model = KeypointDetector(heatmap_sigma=6, maximal_gt_keypoint_pixel_distances="2 4", backbone=backbone_model,
                                minimal_keypoint_extraction_pixel_distance=2, learning_rate=3e-4,
                                keypoint_channel_configuration=confs.joints_name, ap_epoch_start=1,
                                ap_epoch_freq=2, lr_scheduler_relative_threshold=0.0, max_keypoints=20)

    early_stopping = RelativeEarlyStopping(monitor="validation/epoch_loss", patience=5, verbose=True, mode="min",
                                           min_relative_delta=float(args["early_stopping_relative_threshold"]), )

    checkpoint_callback = ModelCheckpoint(monitor="checkpointing_metrics/valmeanAP", mode="max", save_weights_only=True,
                                          save_top_k=1)

    # Total parameters and trainable parameters.
    total_params_1 = sum(p.numel() for p in my_model.parameters())
    print(f"{total_params_1:,} total parameters.")

    total_trainable_params_1 = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f"{total_trainable_params_1:,} training parameters.")

    # Enhanced optimizer and scheduler with warmup and cosine annealing
    optimizer = optim.AdamW(my_model.parameters(), lr=confs.init_lr, weight_decay=1e-4)

    # Cosine annealing with warm restarts for better convergence
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-6, verbose=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-6)
    #
    # calculate steps per epoch for training and test set
    trainSteps = len(train_loader)
    valSteps = len(val_loader)

    # initialize a dictionary to store training historyanlysis the full code base and explainn it in debuging style and dose the code use pretraind hrnet by default  
    H = {"train_loss": [], "val_loss": []}

    # To monitor training loss
    train_loss_hist = Averager()

    # To store training loss and mAP values.
    train_loss_list = []

    map_50_list = []
    map_list = []

    # To save best model.
    save_best_model = SaveBestModel()
    
    # Variables for resuming training
    start_epoch = 0
    
    # Load checkpoint if specified
    if args.get('resume_checkpoint'):
        try:
            checkpoint_info = load_checkpoint(
                args['resume_checkpoint'], 
                my_model, 
                optimizer, 
                scheduler
            )
            start_epoch = checkpoint_info['epoch']
            save_best_model.best_valid_map = checkpoint_info['best_valid_map']
            print(f"Resuming training from epoch {start_epoch}")
            print(f"Best validation mAP so far: {save_best_model.best_valid_map:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for epoch in range(start_epoch, confs.epochs):
        print(f"\nEpoch {epoch + 1} of {confs.epochs}")

        # Reset the training loss histories for the current epoch.
        train_loss_hist.reset()

        # Start timer and carry out training and validation.
        start = time.time()

        train_loss = train(my_model)

        metric_summary = validate(my_model)
            # after metric_summary is computed
        save_best_model(
            model=my_model,
            current_valid_map=metric_summary["map"],  # or "map_50" if you prefer
            epoch=epoch,
            out_dir=confs.base_output,
            optimizer=optimizer,
            scheduler=scheduler
        )

        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch + 1} mAP@0.50:0.95: {metric_summary['map']}")
        print(f"Epoch #{epoch + 1} mAP@0.50: {metric_summary['map_50']}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])
        scheduler.step()

    print('TRAINING COMPLETE')




