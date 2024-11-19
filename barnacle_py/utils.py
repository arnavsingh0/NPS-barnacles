"""
========================================================
                    Utils.py
========================================================

Author: Arnav Singh
Date: 2024-11-19
Version: 1.0.0
Description:
    This is the utils for the barnacles code, attempting to simplify the code presented in the
    jupyter notebook. 

Usage:
    detect_green_frame(image_path: str) -> tuple:
        Improved frame detection with enhanced error handling
    
    preprocess_images():
        Process images and masks with proper alignment
    
    create_image_grid(image, grid_size):
        Divide image into grid segments
    
    prepare_datasets():
        Create dataset splits with proper validation
    
    class BarnacleDataset(Dataset):
        A custom dataset class for loading barnacle images and masks

Dependencies:
    numpy, torch, cv2, matplotlib, os, shutil, torch.utils.data, torchvision.transforms

License:
    MIT License

========================================================
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
import shutil

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Suppress PNG warnings
os.environ['OPENCV_IO_IGNORE_WARNINGS'] = '1'

def detect_green_frame(image_path: str) -> tuple:
    """Improved frame detection with enhanced error handling"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # HSV color space conversion
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 10])
        upper_green = np.array([100, 255, 255])
        
        # Mask creation and processing
        mask = cv2.bitwise_not(cv2.inRange(hsv, lower_green, upper_green))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Contour processing
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.9 < aspect_ratio < 1.1 and (w * h) < 6000000:
                valid_contours.append((cv2.contourArea(contour), (x, y, w, h)))
        
        if not valid_contours:
            raise ValueError("No valid green frame detected")
            
        # Select largest valid contour
        _, (x, y, w, h) = max(valid_contours, key=lambda x: x[0])
        return image, (x, y, w, h)
    
    except Exception as e:
        print(f"Error in detect_green_frame: {str(e)}")
        return None, (0, 0, 0, 0)

def preprocess_images():
    """Process images and masks with proper alignment"""
    os.makedirs("processed", exist_ok=True)
    
    # Process original images first
    for img_file in ['img1.png', 'img2.png', 'unseen_img1.png']:
        img_path = os.path.join("Barnacles", img_file)
        image, coords = detect_green_frame(img_path)
        
        if image is not None:
            x, y, w, h = coords
            # Save cropped image
            cv2.imwrite(os.path.join("processed", img_file), image[y:y+h, x:x+w])
            
            # Process corresponding mask with same coordinates
            mask_file = img_file.replace("img", "mask")
            mask_path = os.path.join("Barnacles", mask_file)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    cropped_mask = mask[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join("processed", mask_file), cropped_mask)

def create_image_grid(image, grid_size):
    """Divide image into grid segments"""
    h, w = image.shape[:2]
    step_x, step_y = w // grid_size[1], h // grid_size[0]

    segments = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x_start = j * step_x
            y_start = i * step_y
            segments.append(image[y_start:y_start+step_y, x_start:x_start+step_x])
    return segments

def prepare_datasets():
    """Create dataset splits with proper validation"""
    os.makedirs("train/data", exist_ok=True)
    os.makedirs("train/mask", exist_ok=True)
    os.makedirs("val/data", exist_ok=True)
    os.makedirs("val/mask", exist_ok=True)
    os.makedirs("test", exist_ok=True)

    # Process all cropped files
    for file in os.listdir("processed"):
        src_path = os.path.join("processed", file)
        
        if 'unseen' in file:
            dest = "test"
        else:
            dest = "train"
        
        # Load and divide image/mask
        image = cv2.imread(src_path)
        segments = create_image_grid(image, (10, 10))
        
        # Save segments
        for i, segment in enumerate(segments):
            if 'mask' in file:
                output_dir = os.path.join(dest, "mask")
            else:
                output_dir = os.path.join(dest, "data")
            
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_seg{i}.png")
            cv2.imwrite(output_path, segment)

    # Create validation split
    train_files = os.listdir("train/data")
    val_files = np.random.choice(train_files, size=int(len(train_files)*0.2), replace=False)
    
    for file in val_files:
        shutil.move(os.path.join("train/data", file), os.path.join("val/data", file))
        shutil.move(os.path.join("train/mask", file.replace("img", "mask")), 
                   os.path.join("val/mask", file.replace("img", "mask")))

class BarnacleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        
        # Separate transform for masks
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Resize((256, 256)),  # Resize
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace("img", "mask"))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)  # Use mask-specific transform
            
        return image, mask

# Main execution flow
if __name__ == "__main__":
    # 1. Preprocess images and masks
    preprocess_images()
    
    # 2. Create dataset splits
    prepare_datasets()
    
    # 3. Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 4. Create datasets
    train_dataset = BarnacleDataset('train/data', 'train/mask', transform)
    val_dataset = BarnacleDataset('val/data', 'val/mask', transform)
    test_dataset = BarnacleDataset('test', 'test', transform) if os.path.exists("test") else None