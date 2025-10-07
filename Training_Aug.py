from __future__ import annotations
import sys
import os
import math
import time
import json
import argparse
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from typing import Dict, List, Tuple
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model.hydranet import HydraFusion  # Your model import

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

np.float = float
np.int = int

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as F_t
# import inspect # Not needed for this version

import logging

# remove any existing handlers on root
for h_idx in range(len(logging.root.handlers[:])):
    h = logging.root.handlers[0]
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------ Global Configuration ------------------------
CLASS_MAP = {
    "car": 1, "van": 2, "truck": 3, "bus": 4,
    "motorbike": 5, "bicycle": 6,
    "pedestrian": 7, "group of pedestrian": 8
}

CLASS_MAP_2 = {
    "c": 1, "v": 2, "t": 3, "bu": 4,
    "m": 5, "bi": 6,
    "p": 7, "gop": 8
}
NUM_CLASSES = len(CLASS_MAP) + 1  # +1 for background


# Add sequence condition mapping
CONDITION_MAP = {
    "good": ["city", "highway", "junction", "night", "motorway", "rural"],
    "bad": ["fog", "snow", "rain"]
}

def insert_clutter_blobs(tensor: torch.Tensor, num: int = 3):
    _, H, W = tensor.shape
    for _ in range(num):
        cx, cy = np.random.randint(50, W-50), np.random.randint(50, H-50)
        r = np.random.randint(5, 15)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        mask = ((xx - cx)**2 + (yy - cy)**2) <= r**2
        tensor[:, mask] += torch.rand(1).item() * 0.5
    return torch.clamp(tensor, 0, 1)


def apply_attenuation(tensor: torch.Tensor, scale=0.5):
    return tensor * scale  # 0.3 to 0.6 simulates fog/rain

def radial_attenuation(tensor: torch.Tensor):
    _, H, W = tensor.shape
    cx, cy = W // 2, H // 2
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = torch.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = dist.max()
    fade = 1.0 - (dist / max_dist) * 0.5  # fade more with distance
    return tensor * fade.to(tensor.device)

# =============================== VALIDATION FUNCTION ===============================
def run_validation(model_path: str, data_root: str, sequences: List[str], 
                   condition: str = "all", augment: bool = False, 
                   batch_size: int = 4, device: torch.device = torch.device("cuda")):
    """
    Run validation on a pre-trained model
    Args:
        model_path: Path to the saved model checkpoint
        data_root: Root directory of RADIATE dataset
        sequences: List of sequence names to validate on
        condition: Weather condition to filter sequences ("good", "bad", or "all")
        augment: Whether to apply augmentation during validation
        batch_size: Batch size for validation
        device: Device to run validation on
    """
    # Load model
    if not Path(model_path).exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return
    
    # Create config (simplified for validation)
    config_dict = {
        "device": device, "num_classes": NUM_CLASSES, "pretrained": False,
        "enable_radar": True, "enable_lidar": True, "enable_camera": True,
        "input_channels": 3, "backbone": "resnet18",
        "feature_levels": [2, 3, 4, 5],
        "anchor_sizes": [[32, 40, 50], [64, 80, 100], [128, 160, 200], [256, 320, 400], [512, 640, 800]],
        "aspect_ratios": [[0.5, 1.0, 2.0]],
    }
    config = SimpleNamespace(**config_dict)
    
    model = HydraFusion(config).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    
    # Prepare dataset
    map_metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=[0.5, 0.75]).to(device)
    
    # Filter sequences by condition
    if condition != "all":
        valid_sequences = CONDITION_MAP.get(condition, [])
        sequences = [seq for seq in sequences if any(good_seq in seq for good_seq in valid_sequences)]
        logger.info(f"Filtered sequences for {condition} condition: {sequences}")
    
    if not sequences:
        logger.error("No sequences found for validation")
        return
    
    # Create validation dataset
    val_datasets = []
    for seq in sequences:
        seq_path = Path(data_root) / seq
        if not seq_path.exists():
            logger.warning(f"Sequence path {seq_path} missing. Skipping.")
            continue
        val_datasets.append(
            HydraFusionDataset(
                seq_path=seq_path,
                validation=True,
                val_split=0.5,  # Use entire sequence
                augment=augment  # Apply augmentation if requested
            )
        )
    
    if not val_datasets:
        logger.error("No valid validation datasets created")
        return
    
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True if device.type != "cpu" else False
    )
    
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Run validation
    avg_val_loss, avg_val_loss_dict, map_results = validate(
        model, val_loader, device, map_metric, SimpleNamespace(overfit_test=False), 
        0, 1  # Dummy epoch numbers
    )
    
    # Print results
    logger.info(f"Validation Results for {condition} conditions:")
    logger.info(f"Loss: {avg_val_loss:.4f}")
    map_50 = map_results.get('map_50', torch.tensor(0.0)).item()
    map = map_results.get('map', torch.tensor(0.0)).item()
    logger.info(f"mAP@0.50: {map_50:.4f}, mAP@0.50:0.95: {map:.4f}")


# ------------------------ Annotation Processing ------------------------
def build_frame_annotations(json_file: Path) -> Dict[int, List[Dict]]:
    if not json_file.exists():
        logger.warning(f"Annotation file not found: {json_file}")
        return {}
    with open(json_file) as f:
        data = json.load(f)
    annotations = defaultdict(list)
    for obj in data:
        label = CLASS_MAP.get(obj["class_name"], -1)
        if label == -1: continue
        for frame_idx, bbox_entry in enumerate(obj["bboxes"]):
            if not bbox_entry: continue
            x, y, w, h = bbox_entry["position"]
            angle = math.radians(bbox_entry["rotation"])
            cx, cy = x + w / 2, y + h / 2
            annotations[frame_idx].append({
                "label": label,
                "bbox": torch.tensor([cx, cy, w, h, angle], dtype=torch.float32)
            })
    logger.info(
        f"Loaded {sum(len(v) for v in annotations.values())} annotations across {len(annotations)} frames from {json_file.name}")
    return annotations


# ---------------------------- Dataset Class -----------------------------
class HydraFusionDataset(Dataset):
    BEV_RES = 0.17361
    BEV_SIZE = 1152
    ORIGIN = BEV_SIZE // 2

    def __init__(self, seq_path: Path, transform=None, validation=False, val_split=0.2, augment = False, viz_mode = False):
        ### MODIFIED ###
        self.augment = augment
        self.viz_mode = viz_mode
        self.validation = validation

        self.camera_transforms = T.Compose([
            T.RandomApply([T.GaussianBlur(kernel_size=(21, 21), sigma=(8.0, 12.0))], p=0.7),  # Extreme blur
            T.ColorJitter(brightness=0.6, contrast=0.2, saturation=0.0, hue=0.4),
            T.RandomHorizontalFlip(p=0.7),  # Desaturate + hue distort
            T.ToTensor(),
            T.RandomApply([T.Lambda(lambda img: torch.clamp(img + torch.randn(1, *img.shape[1:]).expand_as(img) * 0.025, 0., 1.))], p=0.7),
            T.ToPILImage()
        ])

        """self.radar_transforms = T.Compose([
            T.ToTensor(),
            T.RandomApply([T.Lambda(lambda img: torch.clamp(img + torch.randn(1, *img.shape[1:]).expand_as(img) * 0.20  , 0., 1.))], p=1),
            T.ToPILImage()
        ])"""

        self.radar_transforms = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda img: apply_attenuation(img, scale=np.random.uniform(0.4, 0.7))),
            T.RandomApply([T.Lambda(lambda img: insert_clutter_blobs(img, num=3))], p=0.5),
            T.RandomApply([T.Lambda(lambda img: radial_attenuation(img))], p=0.5),
            T.Lambda(lambda img: torch.clamp(img + torch.randn_like(img) * 0.05, 0., 1.0)),  # noise
            T.ToPILImage()
        ])

        # For visualization
        self.class_names = {v: k for k, v in CLASS_MAP_2.items()}
        self.class_colors = {
            1: 'red', 2: 'blue', 3: 'green', 4: 'orange',
            5: 'purple', 6: 'yellow', 7: 'cyan', 8: 'magenta'
        }

        ### MODIFIED ###

        self.seq_dir = seq_path
        self.transform = transform
        if not self.seq_dir.exists(): raise FileNotFoundError(f"Sequence directory {self.seq_dir} does not exist")
        self.timestamps = {}
        for sensor, filename in {"left": "zed_left.txt", "right": "zed_right.txt", "radar": "Navtech_Cartesian.txt",
                                 "lidar": "velo_lidar.txt"}.items():
            sensor_path = self.seq_dir / filename
            if not sensor_path.exists(): self.timestamps[sensor] = []; logger.warning(
                f"Missing sensor timestamps: {sensor_path}"); continue
            self.timestamps[sensor] = self._load_timestamps(sensor_path)
        ann_path = self.seq_dir / "annotations" / "annotations.json"
        self.annotations = build_frame_annotations(ann_path)
        all_samples = self._create_samples(is_training=not validation)

        if val_split > 0 and val_split < 1 and len(all_samples) > 1:
            split_idx = int(len(all_samples) * (1 - val_split))
            if validation:
                self.samples = all_samples[split_idx:]
            else:
                self.samples = all_samples[:split_idx]
        elif validation:
            self.samples = all_samples
        else:
            self.samples = all_samples

        # Infer condition from sequence name
        seq_name = seq_path.name.lower()
        print(f"SEQ {seq_name}")
        if any(bad_seq in seq_name for bad_seq in CONDITION_MAP["bad"]):
            self.original_length = len(self.samples)
            self.samples = self.samples * 2  # Duplicate samples
            self.condition = "bad"
        elif any(good_seq in seq_name for good_seq in CONDITION_MAP["good"]):
            self.condition = "good"
        else:
            self.condition = "unknown"  # fallback


        # Disable augmentation if bad condition
        self.augment = True if self.condition == "bad" else False
        
        logger.info(
            f"Loaded {len(self.samples)} samples for {'validation' if validation else 'training'} from {seq_path.name} (val_split={val_split})")

    def _load_timestamps(self, path: Path) -> List[Tuple[int, float]]:
        timestamps = []
        try:
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            frame_idx, timestamp = int(parts[1]), float(parts[3]); timestamps.append(
                                (frame_idx, timestamp))
                        except ValueError:
                            logger.warning(f"Skipping malformed line in {path}: {line.strip()}")
        except Exception as e:
            logger.error(f"Error loading timestamps from {path}: {e}")
        return timestamps

    def _create_samples(self, is_training: bool) -> List[Dict]:
        samples = []
        if not self.timestamps.get("radar"):
            logger.error(f"Cannot create samples for {self.seq_dir.name}: radar timestamps missing or key not found.")
            return samples
        for radar_idx, radar_ts in self.timestamps["radar"]:
            if is_training and len(self.annotations) > 0 and radar_idx not in self.annotations:
                continue
            samples.append({"radar_idx": radar_idx,
                            "left_idx": self._find_nearest(radar_ts, "left"),
                            "right_idx": self._find_nearest(radar_ts, "right"),
                            "lidar_idx": self._find_nearest(radar_ts, "lidar"),
                            "annotations": self.annotations.get(radar_idx, [])})
        return samples
    
    def visualize_sample(self, sample_idx, model=None, device=None, ground_truth=True, save_path=None):
        sample = self[sample_idx]
        if self.viz_mode:
            inputs, targets, sample_info = sample
        else:
            inputs, targets = sample
            sample_info = self.samples[sample_idx]

        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f"Sample {sample_idx} - Frame {sample_info['radar_idx']}", fontsize=16)

        # Camera images
        left_img = (inputs['leftcamera_x'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        right_img = (inputs['rightcamera_x'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Radar & LiDAR
        radar_img = (inputs['radar_x'][0].numpy() * 255).astype(np.uint8)
        lidar_img = (inputs['bev_lidar_x'][0].numpy() * 255).astype(np.uint8)

        axs[0, 0].imshow(left_img); axs[0, 0].set_title("Left Camera"); axs[0, 0].axis('off')
        axs[0, 1].imshow(right_img); axs[0, 1].set_title("Right Camera"); axs[0, 1].axis('off')
        axs[1, 0].imshow(radar_img, cmap='viridis'); axs[1, 0].set_title("Radar BEV")
        axs[1, 1].imshow(lidar_img, cmap='hot'); axs[1, 1].set_title("LiDAR BEV")

        # Get boxes/labels
        if ground_truth:
            boxes = targets['boxes'].cpu().numpy()
            labels = targets['labels'].cpu().numpy() + 1
        else:
            if model is None or device is None:
                logger.warning("Model and device must be provided for predicted box visualization.")
                return
            model.eval()
            with torch.no_grad():
                inputs_on_device = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
                _, detections = model(**inputs_on_device, radar_y=[targets], cam_y=[targets])
            dets = detections.get('camera_both', [])
            if dets and isinstance(dets, list) and len(dets) > 0:
                det = dets[0]
                boxes = det['boxes'].cpu().numpy() if 'boxes' in det else np.zeros((0, 4))
                labels = det['labels'].cpu().numpy() + 1 if 'labels' in det else np.zeros((0,), dtype=int)
            else:
                boxes = np.zeros((0, 4)); labels = np.zeros((0,), dtype=int)

        # Draw boxes on BEV images
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            for ax in [axs[1, 0], axs[1, 1]]:
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        linewidth=2, edgecolor=self.class_colors.get(label, 'white'), facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, self.class_names.get(label, 'unknown'),
                        color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()



    def visualize_right_camera_augmentation(self, sample_idx, save_path=None):
        """Visualize original vs augmented right camera image for comparison"""
        sample = self.samples[sample_idx]
        
        # Load original image
        orig_img = None
        if sample["right_idx"] >= 0:
            p = self.seq_dir / "zed_right" / f"{sample['right_idx']:06d}.png"
            if p.exists():
                orig_img = Image.open(p).convert("RGB")
        
        # Load augmented image
        aug_img = None
        if sample["right_idx"] >= 0:
            p = self.seq_dir / "zed_right" / f"{sample['right_idx']:06d}.png"
            if p.exists():
                aug_img = Image.open(p).convert("RGB")
                if self.augment:
                    aug_img = self.camera_transforms(aug_img)
        
        if orig_img is None or aug_img is None:
            logger.warning(f"Could not load images for sample {sample_idx}")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle(f"Right Camera Augmentation - Sample {sample_idx}", fontsize=16)
        
        # Show original
        ax1.imshow(orig_img)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Show augmented
        ax2.imshow(aug_img)
        ax2.set_title("Augmented Image")
        ax2.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_radar_augmentation(self, sample_idx: int, save_path: Path = None):
        """Visualize original vs augmented radar BEV image for comparison"""
        sample = self.samples[sample_idx]

        orig_img = None
        aug_img = None

        if sample["radar_idx"] >= 0:
            p = self.seq_dir / "Navtech_Cartesian" / f"{sample['radar_idx']:06d}.png"
            if p.exists():
                radar_img = Image.open(p).convert("L")

                orig_img = np.array(radar_img)
                if self.augment and self.radar_transforms:
                    aug_tensor = self.radar_transforms(radar_img)
                    if isinstance(aug_tensor, torch.Tensor):
                        aug_img = (aug_tensor[0].numpy() * 255).astype(np.uint8)
                    else:
                        aug_img = np.array(aug_tensor)
                else:
                    aug_img = orig_img.copy()

        if orig_img is None or aug_img is None:
            logger.warning(f"Could not load radar image for sample {sample_idx}")
            return

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle(f"Radar Augmentation - Sample {sample_idx}", fontsize=16)

        ax1.imshow(orig_img, cmap='viridis')
        ax1.set_title("Original Radar BEV")
        ax1.axis('off')

        ax2.imshow(aug_img, cmap='viridis')
        ax2.set_title("Augmented Radar BEV")
        ax2.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    # Add visualization method
    def visualize_sample(self, sample_idx, save_path=None):
        sample = self[sample_idx]
        if self.viz_mode:
            inputs, targets, sample_info = sample
        else:
            inputs, targets = sample
            sample_info = self.samples[sample_idx]
        
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f"Sample {sample_idx} - Frame {sample_info['radar_idx']}", fontsize=16)
        
        # Denormalize camera images
        left_img = (inputs['leftcamera_x'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        right_img = (inputs['rightcamera_x'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Get BEV images (no denormalization needed)
        radar_img = (inputs['radar_x'][0].numpy() * 255).astype(np.uint8)
        lidar_img = (inputs['bev_lidar_x'][0].numpy() * 255).astype(np.uint8)

        # Visualize left camera
        #left_img = inputs['leftcamera_x'].permute(1, 2, 0).numpy()
        axs[0, 0].imshow(left_img)
        axs[0, 0].set_title("Left Camera")
        axs[0, 0].axis('off')
        
        # Visualize right camera
        #right_img = inputs['rightcamera_x'].permute(1, 2, 0).numpy()
        axs[0, 1].imshow(right_img)
        axs[0, 1].set_title("Right Camera")
        axs[0, 1].axis('off')
        
        # Visualize radar
        #radar_img = inputs['radar_x'][0].numpy()  # Use first channel
        axs[1, 0].imshow(radar_img, cmap='viridis')
        axs[1, 0].set_title("Radar BEV")
        
        # Visualize lidar
        #lidar_img = inputs['bev_lidar_x'][0].numpy()  # Use first channel
        axs[1, 1].imshow(lidar_img, cmap='hot')
        axs[1, 1].set_title("LiDAR BEV")
        
        # Draw bounding boxes on BEV views
        boxes = targets['boxes'].numpy()
        labels = targets['labels'].numpy() + 1
        
        # Draw on radar
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin,
                linewidth=2, edgecolor=self.class_colors.get(label, 'white'),
                facecolor='none'
            )
            axs[1, 0].add_patch(rect)
            axs[1, 0].text(
                xmin, ymin, self.class_names.get(label, 'unknown'),
                color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5)
            )
        
        # Draw on lidar (same boxes)
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin,
                linewidth=2, edgecolor=self.class_colors.get(label, 'white'),
                facecolor='none'
            )
            axs[1, 1].add_patch(rect)
            axs[1, 1].text(
                xmin, ymin, self.class_names.get(label, 'unknown'),
                color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5)
            )
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _find_nearest(self, target_ts: float, sensor: str) -> int:
        if not self.timestamps.get(sensor): return -1
        if not self.timestamps[sensor]: return -1
        return min(self.timestamps[sensor], key=lambda x: abs(x[1] - target_ts))[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        left = torch.zeros((3, 376, 672), dtype=torch.float32)
        right = torch.zeros((3, 376, 672), dtype=torch.float32)
        radar = torch.zeros((3, self.BEV_SIZE, self.BEV_SIZE), dtype=torch.float32)
        bev = torch.zeros((3, self.BEV_SIZE, self.BEV_SIZE), dtype=torch.float32)

        apply_augment = self.augment
        if self.condition == "bad" and not self.validation and self.augment:
            # First half: augmented, second half: non-augmented
            apply_augment = (idx < self.original_length)

        if sample["left_idx"] >= 0:
            p = self.seq_dir / "zed_left" / f"{sample['left_idx']:06d}.png"
            if p.exists(): left = self._load_image(p, apply_transform = apply_augment)
        if sample["right_idx"] >= 0:
            p = self.seq_dir / "zed_right" / f"{sample['right_idx']:06d}.png"
            if p.exists(): right = self._load_image(p, apply_transform= apply_augment)
        if sample["radar_idx"] >= 0:
            p = self.seq_dir / "Navtech_Cartesian" / f"{sample['radar_idx']:06d}.png"
            if p.exists(): radar = self._load_radar(p,transform = apply_augment)
        if sample["lidar_idx"] >= 0:
            p = self.seq_dir / "velo_lidar" / f"{sample['lidar_idx']:06d}.csv"
            bev = self._load_lidar_bev(p)

        target_h, target_w = left.shape[1], left.shape[2]

        ### MODIFIED ###
        scale_x = target_w / self.BEV_SIZE
        scale_y = target_h / self.BEV_SIZE
        ### MODIFIED ###

        if radar.shape[1:] != (target_h, target_w): radar = F.interpolate(radar.unsqueeze(0), size=(target_h, target_w),
                                                                          mode="bilinear", align_corners=False).squeeze(
            0)
        if bev.shape[1:] != (target_h, target_w): bev = F.interpolate(bev.unsqueeze(0), size=(target_h, target_w),
                                                                      mode="bilinear", align_corners=False).squeeze(0)

        fwd_bev = bev.clone()
        inputs = {"leftcamera_x": left, "rightcamera_x": right, "radar_x": radar, "bev_lidar_x": bev,
                  "r_lidar_x": fwd_bev}

        if self.transform: pass

        raw_anns = sample["annotations"];
        boxes_list, labels_list = [], []
        for ann in raw_anns:
            cx, cy, w, h, angle_rad = ann["bbox"]
            xmin, ymin, xmax, ymax = (cx - w / 2) * scale_x, (cy - h / 2)* scale_y, (cx + w / 2) * scale_x, (cy + h / 2) * scale_y
            boxes_list.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32))
            labels_list.append(ann["label"])

        
        if boxes_list:
            boxes4 = torch.stack(boxes_list)
            labels = torch.tensor(labels_list, dtype=torch.int64) - 1
        else:
            boxes4 = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)

        targets = {"boxes": boxes4, "labels": labels}

        if self.viz_mode:
            return inputs, targets, sample  # Return additional sample info for visualization
        
        return inputs, targets

    """def _load_image(self, path: Path) -> torch.Tensor:
        try:
            return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            logger.warning(f"Error loading image {path}: {e}"); return torch.zeros((3, 376, 672), dtype=torch.float32)"""
    
    # Update _load_image to apply transforms
    def _load_image(self, path: Path, apply_transform=False) -> torch.Tensor:
        try:
            img = Image.open(path).convert("RGB")
            
            if apply_transform and self.camera_transforms and not self.viz_mode:
                img = self.camera_transforms(img)
                
            return F_t.to_tensor(img)
        except Exception as e:
            logger.warning(f"Error loading image {path}: {e}")
            return torch.zeros((3, 376, 672), dtype=torch.float32)
        
        

    """def _load_radar(self, path: Path) -> torch.Tensor:
        try:
            return torch.from_numpy(np.array(Image.open(path).convert("L"))).unsqueeze(0).repeat(3, 1,
                                                                                                 1).float() / 255.0
        except Exception as e:
            logger.warning(f"Error loading radar {path}: {e}"); return torch.zeros((3, self.BEV_SIZE, self.BEV_SIZE),
                                                                                   dtype=torch.float32)"""

    def _load_radar(self, path: Path, transform=None) -> torch.Tensor:
        try:
            radar_img = Image.open(path).convert("L")
            if transform:
                radar_img = self.radar_transforms(radar_img)

            radar_tensor = torch.from_numpy(np.array(radar_img)).unsqueeze(0).repeat(3, 1, 1).float() / 255.0

            return radar_tensor
        
        except Exception as e:
            logger.warning(f"Error loading radar {path}: {e}")
            return torch.zeros((3, self.BEV_SIZE, self.BEV_SIZE), dtype=torch.float32)


    def _load_lidar_bev(self, path: Path) -> torch.Tensor:
        bev = torch.zeros((1, self.BEV_SIZE, self.BEV_SIZE), dtype=torch.float32)
        actual_path = path if path.exists() else path.with_suffix(".txt")
        if not actual_path.exists(): return bev.repeat(3, 1, 1)
        try:
            with open(actual_path) as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 4: continue
                    x, y, intensity = float(parts[0]), float(parts[1]), float(parts[3])
                    px, py = int(round(x / self.BEV_RES)) + self.ORIGIN, self.ORIGIN - int(round(y / self.BEV_RES))
                    if 0 <= px < self.BEV_SIZE and 0 <= py < self.BEV_SIZE: bev[0, py, px] = max(bev[0, py, px].item(),
                                                                                                 intensity / 255.0)
        except Exception as e:
            logger.warning(f"Error loading LiDAR {actual_path}: {e}")
        return bev.repeat(3, 1, 1)


# ---------------------------- Training Setup --------------------------
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    collated_inputs = {k: torch.stack([x[k] for x in inputs]) for k in inputs[0].keys()}
    return collated_inputs, targets


def _aggregate_loss(loss_dict: dict) -> torch.Tensor:
    total = torch.tensor(0.0, device='cpu')
    first_tensor_device = None
    for v_outer in loss_dict.values():
        if isinstance(v_outer, torch.Tensor):
            first_tensor_device = v_outer.device; break
        elif isinstance(v_outer, dict):
            for v_inner in v_outer.values():
                if isinstance(v_inner, torch.Tensor): first_tensor_device = v_inner.device; break
            if first_tensor_device: break
    if first_tensor_device: total = torch.tensor(0.0, device=first_tensor_device)
    for v in loss_dict.values():
        if isinstance(v, dict):
            for subv in v.values():
                if isinstance(subv, torch.Tensor): total = total + subv
        elif isinstance(v, torch.Tensor):
            total = total + v
    return total


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    model.train();
    epoch_loss = 0.0;
    start_time = time.time();
    num_batches = len(dataloader);
    loss_dict_sum = defaultdict(float)
    if num_batches == 0: logger.warning(
        f"Epoch {epoch}: Dataloader is empty. Skipping training for this epoch."); return 0.0, {}
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = {k: v.to(device) for k, v in inputs.items()};
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict_returned, _ = model(**inputs, radar_y=targets, cam_y=targets)
        if not isinstance(loss_dict_returned, dict): logger.error(
            f"Model did not return loss dict. Got: {type(loss_dict_returned)}"); continue
        loss = _aggregate_loss(loss_dict_returned)
        if torch.isnan(loss) or torch.isinf(loss): logger.warning(f"NaN/Inf loss: {loss.item()}. Skipping."); continue
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()
        current_loss_val = loss.item();
        epoch_loss += current_loss_val
        for k, v_loss in loss_dict_returned.items():
            if isinstance(v_loss, dict):
                for subk, subv_loss in v_loss.items():
                    if isinstance(subv_loss, torch.Tensor): loss_dict_sum[f"{k}/{subk}"] += subv_loss.item()
            elif isinstance(v_loss, torch.Tensor):
                loss_dict_sum[k] += v_loss.item()
        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1: logger.info(
            f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx + 1}/{num_batches}] Loss: {current_loss_val:.4f}")
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0;
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict_sum.items()} if num_batches > 0 else {};
    epoch_time = time.time() - start_time
    loss_str_parts = [f"{k}: {v_avg_loss:.4f}" for k, v_avg_loss in avg_loss_dict.items()]
    loss_str = " ".join(loss_str_parts)
    logger.info(
        f"Epoch [{epoch}/{total_epochs}] Average Training Loss: {avg_loss:.4f} ({loss_str}) Time: {epoch_time:.2f}s")
    return avg_loss, avg_loss_dict


def validate(model, dataloader, device, map_metric: MeanAveragePrecision, cli_args, current_epoch_num,
             total_epochs_for_log):
    original_training_state = model.training
    model.train()  # Model needs train() mode for its forward to return losses and detections

    val_loss = 0.0;
    val_loss_dict_sum = defaultdict(float);
    num_batches = len(dataloader);
    map_metric.reset()

    DETAILED_LOG_ACTIVE = False
    if cli_args.overfit_test and current_epoch_num <= 2: DETAILED_LOG_ACTIVE = True  # Log for first 2 overfit epochs

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):  # targets is List[Dict[str, Tensor]]
            inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

            detections_raw_from_model = None;
            loss_dict_returned = {}
            try:
                loss_dict_returned, detections_raw_from_model = model(**inputs_on_device, radar_y=targets_on_device,
                                                                      cam_y=targets_on_device)
            except Exception as e:
                logger.error(
                    f"Exception during model forward in validation (batch {batch_idx}, epoch {current_epoch_num}): {e}",
                    exc_info=True)
                loss_dict_returned = {};
                detections_raw_from_model = []

            if loss_dict_returned and isinstance(loss_dict_returned, dict):
                loss = _aggregate_loss(loss_dict_returned)
                if isinstance(loss, torch.Tensor): val_loss += loss.item()
                for k_loss, v_loss in loss_dict_returned.items():
                    if isinstance(v_loss, dict):
                        for subk, subv_loss in v_loss.items():
                            if isinstance(subv_loss, torch.Tensor): val_loss_dict_sum[
                                f"{k_loss}/{subk}"] += subv_loss.item()
                    elif isinstance(v_loss, torch.Tensor):
                        val_loss_dict_sum[k_loss] += v_loss.item()

            if DETAILED_LOG_ACTIVE and batch_idx == 0:  # Log for first batch if detailed logging is active
                logger.info(
                    f"--- DETAILED LOG: Epoch {current_epoch_num}/{total_epochs_for_log}, Val Batch {batch_idx} ---")
                logger.info(f"Raw 'detections_raw_from_model' (type: {type(detections_raw_from_model)}):")
                if isinstance(detections_raw_from_model, dict):
                    for key, value in detections_raw_from_model.items():
                        first_item_info = "N/A";
                        list_len_info = "N/A"
                        if isinstance(value, list):
                            list_len_info = len(value)
                            if len(value) > 0:
                                first_item_info = f"type {type(value[0])}"
                                if isinstance(value[0], dict):
                                    first_item_info += f", keys: {list(value[0].keys())}"
                                    if 'boxes' in value[0] and isinstance(value[0]['boxes'], torch.Tensor):
                                        first_item_info += f", boxes shape: {value[0]['boxes'].shape}"
                        logger.info(f"  Key '{key}': list of {list_len_info} items. First item info: {first_item_info}")
                elif isinstance(detections_raw_from_model, list):
                    logger.info(
                        f"  Is a list of {len(detections_raw_from_model)} items. First item type: {type(detections_raw_from_model[0]) if detections_raw_from_model else 'N/A'}")
                else:
                    logger.info(f"  Content: {detections_raw_from_model}")

            dets_batch_list_for_map = []
            if isinstance(detections_raw_from_model, dict):
                candidate_list = detections_raw_from_model.get('camera_both')
                if not isinstance(candidate_list, list):
                    if candidate_list is not None: logger.debug(
                        f"Detections for 'camera_both' not a list (type: {type(candidate_list)}) for batch {batch_idx}, epoch {current_epoch_num}. Trying other values.")
                    found_list_in_dict = False
                    for key, val_det_list in detections_raw_from_model.items():
                        if isinstance(val_det_list, list):
                            logger.debug(
                                f"Using detections from key '{key}' in model output dictionary for batch {batch_idx}, epoch {current_epoch_num}.")
                            candidate_list = val_det_list;
                            found_list_in_dict = True;
                            break
                    if not found_list_in_dict and detections_raw_from_model: logger.warning(
                        f"Model output dict did not contain a list of detections for batch {batch_idx}, epoch {current_epoch_num}. Keys: {list(detections_raw_from_model.keys())}")
                if isinstance(candidate_list, list):
                    dets_batch_list_for_map = candidate_list
                elif detections_raw_from_model:
                    logger.warning(
                        f"Could not extract a valid list of detections from model output dict for batch {batch_idx}, epoch {current_epoch_num}.")
            elif isinstance(detections_raw_from_model, list):
                dets_batch_list_for_map = detections_raw_from_model
            elif detections_raw_from_model is not None:
                logger.warning(
                    f"Model 'detections_raw_from_model' is unexpected type: {type(detections_raw_from_model)} for batch {batch_idx}, epoch {current_epoch_num}.")

            if DETAILED_LOG_ACTIVE and batch_idx == 0:
                logger.info(f"Selected 'dets_batch_list_for_map' for mAP (length {len(dets_batch_list_for_map)}):")
                _d0_map = dets_batch_list_for_map[0] if dets_batch_list_for_map and len(
                    dets_batch_list_for_map) > 0 else None
                if _d0_map and isinstance(_d0_map, dict):
                    logger.info(
                        f"  First item of dets_batch_list_for_map (type {type(_d0_map)}): {list(_d0_map.keys())}")
                elif _d0_map:
                    logger.info(f"  First item of dets_batch_list_for_map (type {type(_d0_map)}): {_d0_map}")
                else:
                    logger.info("  dets_batch_list_for_map is empty or items are not dicts.")

            preds_for_metric = [];
            gts_for_metric = []
            num_samples_in_batch = len(targets)

            for i in range(num_samples_in_batch):
                tgt_item = targets[i]
                gts_for_metric.append({'boxes': tgt_item['boxes'].cpu(), 'labels': tgt_item['labels'].cpu()})

                current_pred_boxes_cpu = torch.empty(0, 4, device='cpu')
                current_pred_scores_cpu = torch.empty(0, device='cpu')
                adjusted_pred_labels_cpu = torch.empty(0, dtype=torch.long, device='cpu')
                raw_model_labels_for_sample = torch.empty(0, dtype=torch.long, device='cpu')
                det_item_for_sample_log = "N/A (no detection item or list too short)"

                if i < len(dets_batch_list_for_map):
                    det_item = dets_batch_list_for_map[i]
                    det_item_for_sample_log = det_item
                    if not isinstance(det_item, dict) or not all(k in det_item for k in ['boxes', 'scores', 'labels']):
                        logger.warning(
                            f"Pred item {i} in batch {batch_idx}, epoch {current_epoch_num} has unexpected structure: {type(det_item)}.")
                    else:
                        pred_boxes_cpu_raw = det_item['boxes'].cpu()
                        pred_scores_cpu_raw = det_item['scores'].cpu()
                        raw_model_labels_for_sample = det_item['labels'].cpu()
                        if raw_model_labels_for_sample.numel() > 0:
                            foreground_mask = raw_model_labels_for_sample > 0
                            adjusted_pred_labels_cpu = raw_model_labels_for_sample[foreground_mask] - 1
                            current_pred_boxes_cpu = pred_boxes_cpu_raw[foreground_mask]
                            current_pred_scores_cpu = pred_scores_cpu_raw[foreground_mask]
                else:
                    logger.debug(
                        f"No detection item from model for sample {i} in batch {batch_idx}, epoch {current_epoch_num} (dets_batch_list_for_map length: {len(dets_batch_list_for_map)}).")

                preds_for_metric.append({'boxes': current_pred_boxes_cpu, 'scores': current_pred_scores_cpu,
                                         'labels': adjusted_pred_labels_cpu})

                if DETAILED_LOG_ACTIVE and batch_idx == 0:
                    logger.info(f"  --- Sample {i} in Batch {batch_idx} (Epoch {current_epoch_num}) ---")
                    logger.info(f"    Raw det_item_for_sample_log: {det_item_for_sample_log}")
                    if isinstance(det_item_for_sample_log, dict):
                        logger.info(
                            f"      Raw Model Boxes ({raw_model_labels_for_sample.shape[0]} preds): {det_item_for_sample_log.get('boxes', 'N/A')[:2]}...")
                        logger.info(f"      Raw Model Scores: {det_item_for_sample_log.get('scores', 'N/A')[:2]}...")
                        logger.info(f"      Raw Model Labels: {raw_model_labels_for_sample[:2]}...")
                    logger.info(
                        f"    Preds for Metric: boxes({current_pred_boxes_cpu.shape[0]}), scores({current_pred_scores_cpu.shape[0]}), labels({adjusted_pred_labels_cpu.shape[0]})")
                    if current_pred_boxes_cpu.numel() > 0: logger.info(
                        f"      Metric Boxes: {current_pred_boxes_cpu[:2]}..."); logger.info(
                        f"      Metric Scores: {current_pred_scores_cpu[:2]}..."); logger.info(
                        f"      Metric Labels (0-indexed): {adjusted_pred_labels_cpu[:2]}...")
                    logger.info(
                        f"    GT for Metric: boxes({gts_for_metric[-1]['boxes'].shape[0]}), labels({gts_for_metric[-1]['labels'].shape[0]})")
                    if gts_for_metric[-1]['boxes'].numel() > 0: logger.info(
                        f"      GT Boxes: {gts_for_metric[-1]['boxes'][:2]}..."); logger.info(
                        f"      GT Labels (0-indexed): {gts_for_metric[-1]['labels'][:2]}...")
                    logger.info(f"  ------------------------------------")

            map_metric.update(preds_for_metric, gts_for_metric)
            if DETAILED_LOG_ACTIVE and batch_idx == 0: logger.info(
                f"--- END DETAILED LOG for Val Batch {batch_idx}, Epoch {current_epoch_num} ---")

    model.train(original_training_state)
    avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0;
    avg_val_loss_dict = {k: v / num_batches for k, v in val_loss_dict_sum.items()} if num_batches > 0 else {}
    map_results_default = {k: torch.tensor(0.0) for k in
                           ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large', 'mar_1', 'mar_10',
                            'mar_100', 'mar_small', 'mar_medium', 'mar_large']};
    map_results = map_results_default.copy()
    try:
        computed_results = map_metric.compute()
        if computed_results:
            for k_map, v_map in computed_results.items():
                if isinstance(v_map, torch.Tensor):
                    map_results[k_map] = v_map
                elif isinstance(v_map, (int, float)):
                    map_results[k_map] = torch.tensor(float(v_map))
    except Exception as e:
        logger.error(f"Error computing mAP (epoch {current_epoch_num}): {e}. Returning default mAP values (0.0).",
                     exc_info=True)
    loss_str_parts = [];
    for k_loss, v_avg_loss in avg_val_loss_dict.items(): loss_str_parts.append(f"{k_loss}: {v_avg_loss:.4f}")
    loss_str = " ".join(loss_str_parts) if loss_str_parts else "N/A"
    logger.info(f"Validation Loss (Epoch {current_epoch_num}): {avg_val_loss:.4f} ({loss_str})")
    map_val, map_50_val = map_results.get('map'), map_results.get('map_50')
    if isinstance(map_val, torch.Tensor) and isinstance(map_50_val, torch.Tensor):
        logger.info(
            f"mAP@.50:.95 (Epoch {current_epoch_num}): {map_val.item():.4f}, mAP@.50 (Epoch {current_epoch_num}): {map_50_val.item():.4f}")
    else:
        logger.info(
            f"mAP values not available or not in expected tensor format (Epoch {current_epoch_num}). map: {map_val}, map_50: {map_50_val}")
    return avg_val_loss, avg_val_loss_dict, map_results


# =============================== MAIN FUNCTION ===============================
def main():
    parser = argparse.ArgumentParser(description="Train HydraFusion on RADIATE")
    parser.add_argument("--radiate-root", type=str, required=True, help="Path to RADIATE dataset root")
    parser.add_argument("--seqs", type=str, default="tiny_foggy", help="Comma-separated list of sequences")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split-ratio", type=float, default=0.2,
                        help="Validation split ratio (0-1) for normal run")
    parser.add_argument("--overfit-test", action="store_true", help="Run overfitting test on a few samples.")
    parser.add_argument("--overfit-samples", type=int, default=2, help="Number of samples for overfitting test.")
    parser.add_argument("--overfit-epochs", type=int, default=50, help="Number of epochs for overfitting test.")

    parser.add_argument("--viz-samples", action="store_true", help="Visualize dataset samples without training")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation for camera images")
    parser.add_argument("--num-viz", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--viz-output", type=str, default="viz_samples", help="Output directory for visualizations")


     # Add Val Arguements
    parser.add_argument("--val-only", action="store_true", help="Run validation only (without training)")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for validation")
    parser.add_argument("--val-condition", type=str, default="all", 
                        choices=["good", "bad", "all"], 
                        help="Weather condition for validation")
    parser.add_argument("--val-augment", action="store_true", 
                        help="Use augmentation during validation")
    parser.add_argument("--train-condition", type=str, default="all", 
                        choices=["good", "bad", "all"], 
                        help="Weather condition for training")



    args = parser.parse_args()

    import random

    # Visualization mode
    if args.viz_samples:
        viz_dir = Path(args.viz_output)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        seq_path = Path(args.radiate_root) / args.seqs.split(",")[0]
        dataset = HydraFusionDataset(
            seq_path=seq_path,
            validation=False,
            val_split=0.0,
            augment=args.augment,
            viz_mode=True
        )
        
        random_indices = [random.randint(0, len(dataset)) for i in range(args.num_viz)]

        #logger.info(f"Visualizing {args.num_viz} samples from {seq_path.name}")
        """for i in random_indices:
            save_path = viz_dir / f"sample_{i}_full.png"
            dataset.visualize_sample(i, save_path=save_path)
            logger.info(f"Saved full visualization: {save_path}")"""
        
        # Visualize right camera augmentations
        
        """for i in random_indices:
            save_path = viz_dir / f"sample_{i}_augmentation.png"
            dataset.visualize_right_camera_augmentation(i, save_path=save_path)
            logger.info(f"Saved augmentation com    parison: {save_path}")
        
        logger.info("Visualization complete. Exiting.")"""

        # Visualize radar augmentations
        for i in random_indices:
            save_path = viz_dir / f"sample_{i}_radar_augmentation.png"
            dataset.visualize_radar_augmentation(i, save_path=save_path)
            logger.info(f"Saved radar augmentation comparison: {save_path}")

        logger.info("Visualization complete. Exiting.")
        return

    if args.val_only:
        if not args.checkpoint:
            logger.error("Checkpoint path is required for validation-only mode")
            return
            
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        # Get all folder names in radiate_root directory as sequences
        sequences = [name for name in os.listdir(args.radiate_root) if os.path.isdir(os.path.join(args.radiate_root, name))]


        run_validation(
            model_path=args.checkpoint,
            data_root=args.radiate_root,
            sequences=sequences,
            condition=args.val_condition,
            augment=args.val_augment,
            batch_size=args.batch_size,
            device=device
        )
        return

    # ... existing device setup code ...

    # Filter sequences by condition for training
    sequences = [s.strip() for s in args.seqs.split(",")]
    if args.train_condition != "all":
        valid_sequences = CONDITION_MAP.get(args.train_condition, [])
        sequences = [seq for seq in sequences if any(good_seq in seq for good_seq in valid_sequences)]
        logger.info(f"Filtered sequences for {args.train_condition} condition: {sequences}")

    if torch.cuda.is_available():
        device = torch.device("cuda");
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps");
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu");
        logger.info("Using CPU device")

    map_metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=[0.5, 0.75]).to(device)
    sequences = [s.strip() for s in args.seqs.split(",")]
    current_epochs = args.epochs
    effective_batch_size = args.batch_size

    # Define the common base configuration arguments
    # This is based on your last working config that led to mAP=0 and good losses.
    # It does NOT include hydranet.py specific debug flags like 'force_camera_both_for_debug' or 'debug_box_score_thresh'
    # as those require hydranet.py to be modified to use them. This version assumes original hydranet.py.
    config_dict = {
        "device": device, "num_classes": NUM_CLASSES, "pretrained": True, "dropout": 0.3, "activation": "relu",
        "enable_radar": True, "enable_lidar": True, "enable_camera": True,
        "bn_momentum": 0.1, "bn_eps": 1e-5, "use_bn": True,
        "use_custom_transforms": False, "create_gate_dataset": False, "fusion_sweep": False,
        "skip_fused_detection": False, "fuse_bev_features": True, "target_for_cameras": "radar", "multitask": True,
        "fusion_method": "hydra",
        "enable_cam_fusion": True,
        "enable_radar_fusion": True,  # Assuming this was intended in your original working config
        "enable_lidar_fusion": True,  # Assuming this was intended
        "enable_cam_lidar_fusion": True,
        "enable_radar_lidar_fusion": True,
        "enable_cam_radar_fusion": True,  # Assuming this was intended
        "enable_all_fusion": True,  # Assuming this was intended
        "radar_camera_fusion": True, "use_early_fusion": True, "use_late_fusion": True,
        "use_hard_fusion": True, "use_soft_fusion": True, "use_attention": True,
        "use_cam_lidar_fusion": True,  # This is somewhat redundant with enable_cam_lidar_fusion
        "use_cam_radar_fusion": True,
        "use_radar_lidar_fusion": True,
        "use_multimodal_fusion": True, "use_fpn": True, "use_deformable": False,
        "use_class_weights": False, "freeze_backbone": False, "freeze_bn": False,
        "input_channels": 3, "backbone": "resnet18",
        "feature_levels": [2, 3, 4, 5],
        "anchor_sizes": [[32, 40, 50], [64, 80, 100], [128, 160, 200], [256, 320, 400], [512, 640, 800]],
        "aspect_ratios": [[0.5, 1.0, 2.0]],
        "verbose": False,
        "overfit_test": args.overfit_test  # Pass the CLI argument status
    }
    config = SimpleNamespace(**config_dict)

    if args.overfit_test:
        logger.info(
            f"--- RUNNING OVERFITTING TEST on {args.overfit_samples} samples for {args.overfit_epochs} epochs ---")
        current_epochs = args.overfit_epochs

        seq_path = Path(args.radiate_root) / sequences[0]
        if not seq_path.exists(): logger.error(f"Seq path for overfit test missing: {seq_path}. Exiting."); return

        temp_dataset = HydraFusionDataset(seq_path=seq_path, validation=False, val_split=0.0)
        if len(temp_dataset) == 0: logger.error(f"No samples from {seq_path} for overfit. Check dataset."); return

        logger.info(
            f"Scanning {len(temp_dataset)} samples in {sequences[0]} to find {args.overfit_samples} with annotations...")
        annotated_indices_all = [i for i, (_, target) in enumerate(temp_dataset) if target['boxes'].numel() > 0]

        if len(annotated_indices_all) < args.overfit_samples:
            logger.warning(
                f"Found only {len(annotated_indices_all)} annotated samples, wanted {args.overfit_samples}. Using available.")
            if not annotated_indices_all: logger.error("No annotated samples for overfit test. Cannot proceed."); return
            selected_indices = annotated_indices_all
        else:
            selected_indices = annotated_indices_all[:args.overfit_samples]

        logger.info(f"Selected indices for overfitting: {selected_indices}")

        overfit_subset = Subset(temp_dataset, selected_indices)

        train_dataset, val_dataset = overfit_subset, overfit_subset
        effective_batch_size = min(args.batch_size, len(overfit_subset))
        if effective_batch_size == 0 and len(overfit_subset) > 0: effective_batch_size = len(overfit_subset)
        logger.info(
            f"Overfitting test setup: Training/Validation on {len(overfit_subset)} samples. Batch size: {effective_batch_size}. Epochs: {current_epochs}")

    else:  # Normal training setup
        val_split_ratio = args.val_split_ratio
        train_datasets, val_datasets = [], []
        for seq in sequences:
            seq_path = Path(args.radiate_root) / seq
            if not seq_path.exists(): logger.error(f"Sequence path {seq_path} missing. Skipping."); continue
            train_datasets.append(HydraFusionDataset(seq_path, validation=False, val_split=val_split_ratio, augment= args.augment))
            val_datasets.append(HydraFusionDataset(seq_path, validation=True, val_split=val_split_ratio, augment= False))

        if not train_datasets: logger.error("No valid training datasets loaded. Exiting."); return

        train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 and val_datasets and val_datasets[
            0] is not None else (val_datasets[0] if val_datasets else None)
        effective_batch_size = args.batch_size
        logger.info(f"Normal run setup. Epochs: {current_epochs}")

    if len(train_dataset) == 0: logger.error("Training dataset is empty. Exiting."); return

    # Determine num_workers based on CPU count
    num_avail_workers = (os.cpu_count() or 1)
    num_workers_to_use = min(4, num_avail_workers - 1) if num_avail_workers > 1 else 0

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=True if device.type != "cpu" else False
    )

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=effective_batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
            pin_memory=True if device.type != "cpu" else False
        )
    else:
        logger.warning("Validation dataset is empty or None. Validation will be skipped.")

    logger.info(
        f"Final Effective Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset) if val_dataset else 0}")
    if len(train_loader) == 0: logger.error("Train loader is empty, cannot train."); return

    model = HydraFusion(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=current_epochs)

    start_epoch = 0;
    best_val_map = 0.0
    logger.info(f"Starting training from epoch {start_epoch + 1} on {device} for {current_epochs} epochs.")
    logger.info(f"Effective batch size: {effective_batch_size}")

    for epoch in range(start_epoch, current_epochs):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, device, epoch + 1, current_epochs)

        if val_loader and len(val_loader) > 0:
            avg_val_loss, _, current_map_results = validate(
                model, val_loader, device, map_metric, args, epoch + 1, current_epochs
            )
            current_map_val = 0.0
            if current_map_results and isinstance(current_map_results.get('map'), torch.Tensor):
                current_map_val = current_map_results['map'].item()
                # In overfitting, any mAP > 0 (and ideally improving) is a sign of life.
                # In normal training, we track the best.
                is_new_best = False
                if args.overfit_test:
                    if current_map_val > best_val_map:  # For overfit, any improvement from 0 is good
                        best_val_map = current_map_val
                        is_new_best = True
                elif current_map_val > best_val_map:  # For normal run
                    best_val_map = current_map_val
                    is_new_best = True

                if is_new_best:
                    logger.info(
                        f"🎉 Epoch {epoch + 1}: New best mAP@.50:.95: {best_val_map:.4f} (Val Loss: {avg_val_loss:.4f})")

            elif args.overfit_test:  # mAP results were not as expected
                logger.info(
                    f"Epoch {epoch + 1} (Overfit Test): Val Loss: {avg_val_loss:.4f}, mAP was 0 or not determined from results.")
        else:
            logger.info(
                f"Epoch {epoch + 1}: Training loss: {train_loss:.4f}. Skipping validation as val_loader is empty/None.")

        scheduler.step()
        # Ensure model is in training mode for the next epoch's training pass,
        # as validate() sets it to model.train() but then restores original state.
        # The train_one_epoch function should already set model.train() at its start.
        # No, validate sets it to original_training_state which was True. So it's fine.
        # model.train() # This should be handled by train_one_epoch

    logger.info("Training completed.")
    logger.info(
        f"{'Overfitting test' if args.overfit_test else 'Normal training'} completed. Final best mAP: {best_val_map:.4f}")


if __name__ == "__main__":
    main()
