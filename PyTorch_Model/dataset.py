import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load images (BGR to RGB) and mask (grayscale)
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        # Normalize and threshold mask
        mask = mask / 255.0
        mask = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)

        # Normalize image to [0,1]
        img = img / 255.0

        # Convert to torch tensors and rearrange axes for image (C,H,W)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # add channel dim

        return img, mask
