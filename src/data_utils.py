import os
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    EnsureTyped,
    Orientationd,
    Spacingd,
    DivisiblePadd  # Added to fix the 86 vs 96 error
)
from monai.apps import DecathlonDataset
from monai.data import DataLoader, list_data_collate # Added list_data_collate

def get_msd_transforms():
    """Returns the 3D preprocessing pipeline for CT images."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # FIX 1: Explicitly set nearest interpolation for labels to keep values as 0, 1, 2
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.5, 1.5, 1.5), 
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # FIX 2: Added allow_smaller=True to prevent crashes on smaller volumes
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=4,
            image_key="image",
            image_threshold=0,
            allow_smaller=True 
        ),
        # FIX 3: Force the final output to be divisible by 32 (96x96x96) 
        # This solves the "spatial shape 86" error for SwinUNETR
        DivisiblePadd(keys=["image", "label"], k=32),
        EnsureTyped(keys=["image", "label"]),
    ])

def get_dataloader(root_dir, batch_size=1):
    """Initializes MSD Liver dataset and returns a DataLoader."""
    os.makedirs(root_dir, exist_ok=True)

    dataset = DecathlonDataset(
        root_dir=root_dir,
        task="Task08_HepaticVessel",
        section="training",
        transform=get_msd_transforms(),
        download=False 
    )

    # FIX 4: Use list_data_collate because RandCropByPosNegLabeld returns 4 samples per image
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep at 0 for initial debugging on the node
        pin_memory=True,
        collate_fn=list_data_collate 
    )
    return loader
