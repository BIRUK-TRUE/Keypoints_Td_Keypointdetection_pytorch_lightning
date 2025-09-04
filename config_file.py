""" avoid circular imports by separating types"""
from typing import List, Tuple

# read training data
# Root directory containing the MS COCO dataset. Expected structure:
#   <base_path>/annotations/person_keypoints_train2017.json
#   <base_path>/annotations/person_keypoints_val2017.json
#   <base_path>/images/train2017/train217/*.jpg
#   <base_path>/images/val2017/val2017/*.jpg
# base_path = '/Datasets/ms_coco'  sampel
base_path = '/kaggle/input/key-point-data/dataset/ms_coco'
# base_path = '../Datasets/ms_coco'

dataset_type = 'ms_coco'  # ['ms_coco', 'mpii', 'lsp']
dataset_category = ['train', 'val']
dataset_phase = ['train2017', 'val2017']  # ['train2017_a', 'val2017_a']
# COCO official person keypoint annotations
ann_path = base_path + '/annotations/person_keypoints_'
ann_type = 'person_keypoints'
# Default training JSON path
joints_def = ann_path + dataset_phase[0] + '.json'
save_model_path = './snapshots/coco/'

all_joints = [[0], [2, 1], [4, 3], [6, 5], [8, 7], [10, 9], [12, 11], [14, 13], [16, 15]]
all_joints_names = ['nose', 'eye', 'ear', 'shoulder', 'elbow', 'hand', 'hip', 'knee', 'foot']
pose_joint_person = '_epoch'  # _pose_dist_epoch, _pose_loc_epoch, _person_epoch, or _joint_epoch
# COCO 17 human keypoints; each sublist defines one detector heatmap channel
joints_name = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
               'left_elbow', 'right_elbow',  'left_wrist',  'right_wrist',  'left_hip', 'right_hip', 'left_knee',
                'right_knee','left_ankle','right_ankle']
# joints_name_relation = [[0, 'nose'], [1, 'left_eye'], [2, 'right_eye'], [3, 'left_ear'], [4, 'right_ear'],
#                         [5, 'left_shoulder'], [6, 'right_shoulder'], [7, 'left_elbow'], [8, 'right_elbow'],
#                         [9, 'left_wrist'], [10, 'right_wrist'], [11, 'left_hip'], [12, 'right_hip'], [13, 'left_knee'],
#                         [14, 'right_knee'], [15, 'left_ankle'], [16, 'right_ankle']]

# Classes: 0 index is reserved for background
bbox_class = ['__background__', 'person']
num_joints = 17
model_type = 'HRNet'  # LightHRNet, unet, 'resnet18, resnet34, resnet50, resnet101, resnet152, CustomCNN', 'PretrainedResnet'
hrnet_pretrained = True  # Set to True to use ImageNet pretrained weights
hrnet_pretrained_path = "pretrained_weights/hrnet_w32.pth"  # Path to local pretrained weights file
base_output = "snapshots/"  # define the path to the base output directory

# Light HRNet backbone configuration - Enhanced for better performance
light_hrnet_channels = 48  # Slightly smaller for better efficiency
light_hrnet_stages = 2
light_hrnet_branches = 4   # More branches for multi-scale features
light_hrnet_blocks = 2

batch_size = 8  # Reduced for human pose estimation stability
dropout_rate = 0.3  # Reduced for better keypoint learning
stride = 4.0  # Smaller stride for better keypoint localization
epochs = 10  # Increased for proper human pose learning
init_lr = 1e-4  # Lower learning rate for stable human keypoint training
optimizer_name = 'adamw'  # sgd, adam, adamw, rmsprop, sdgard
# mean_pixel = [123.68, 116.779, 103.939]
mean_pixel = [0.485, 0.456, 0.406]
std_pixel = [0.229, 0.224, 0.225]
shuffle = True
mirror = True
crop = False

# img_size = 480  # 640, 480, 320, 256  224  Image size to resize to in transforms
img_depth = 3  # 3, 1  the number of channels in the input
img_width = 256 # 512
img_height = 256  # 480
min_input_size = 100
img_max_width = 640
img_max_height = 640
num_main_pairing = 6
max_people = 20
max_num_joints = 14
max_joints_val = max(img_max_height - 1, img_max_width - 1)
max_bbox_coords_val = max(615.41, 625.57)
max_bbox_length = max(img_max_height, img_max_width)

num_workers = 2
random_state = 42  # To shuffle images and targets in same order
threshold = 0.5  # define threshold to filter weak predictions

image_indices = 0
curr_train_img_idx = 0
val_image_indices = 0
curr_val_img_idx = 0
save_iters = 10
display_iters = 20

alb_img_height = 240  # 512, 480, 448, 384, 320, 240
alb_img_width = 320  # 640, 576, 512, 448, 400, 320

intermediate_supervision = True
intermediate_supervision_layer = 12

global_scale = 1.0
pos_dist_thresh = 13  # 19, 17, 13, 11, 7
scale_jitter_lo = 0.85
scale_jitter_up = 1.15

coords_eval = True
coords_huber_loss = True
coords_loss_weight = 1.0  # 0.05
coords_stdev = 7.2801  # 7.2801

pairing_eval = True
pairing_huber_loss = True
pairing_loss_weight = 1.0  # 0.05
collect_pairing_stats = False
pairing_stats_fn = './libs/coco_pairing/pairing_stats.mat'
pairing_model_dir = './libs/coco_pairing/'
tensorflow_pairing_order = True
coco_categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                   "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                   "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                   "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                   "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                   "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                   "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

bbox_eval = True
bbox_huber_loss = True
bbox_loss_weight = 0.05  # 0.05
bbox_mean = [0.0, 0.0, 0.0, 0.0]
bbox_stdev = [0.1, 0.1, 0.2, 0.2]
bbox_nms = 0.5  # non-maximum suppression threshold

regularize = False
weight_decay = 0.0001  # regularization 0.00001 or 0.0001 or 5e-4
channels_dim = -1  # initialize the input shape to be "channels last" and the channels dimension itself
bn_eps = 0.001  # 2e-5,  0.001
bn_momentum = 0.99  # 0.99, 0.9
weigh_part_predictions = False
weigh_negatives = False
fg_fraction = 0.25
weigh_only_present_joints = False
l2_val = 0.0001

crop_pad = 0
scoremap_dir = "test"
dataset = ""
use_gt_segm = False

video = False
video_batch = False
sparse_graph = []

KEYPOINT_TYPE = Tuple[int, int]  # (u,v)
COCO_KEYPOINT_TYPE = Tuple[int, int, int]  # (u,v,f)
CHANNEL_KEYPOINTS_TYPE = List[KEYPOINT_TYPE]
IMG_KEYPOINTS_TYPE = List[CHANNEL_KEYPOINTS_TYPE]

