from typing import Any

from glob import glob
from pathlib import Path

import albumentations
import cv2
import numpy as np
from torch.utils.data import Dataset


def byte2arr(img_bytes: Any, to_float32: bool = False) -> np.ndarray:
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_arr = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    if to_float32:
        img_arr = img_arr.astype(np.float32) / 255.
    return img_arr


class CustomDataset(Dataset):
    def __init__(self, mode: str = "train", **kwargs):
        self.kwargs = kwargs
        self.mode = mode
        self.hr_file_list = glob(f"{self.kwargs['dataset_dir']}/{mode}/hr/*{kwargs['data_endswith']}")
        self.lr_file_list = glob(f"{self.kwargs['dataset_dir']}/{mode}/lr_x{self.kwargs['scale_factor']}/*{kwargs['data_endswith']}")

    def __len__(self):
        return len(self.hr_file_list)

    def __getitem__(self, item):
        hr_img_file, lr_img_file = self.hr_file_list[item], self.lr_file_list[item]
        with (open(hr_img_file, 'rb') as f1, open(lr_img_file, 'rb') as f2):
            hr_bytes, lr_bytes = f1.read(), f2.read()
            hr_data,  lr_data = byte2arr(hr_bytes), byte2arr(lr_bytes)
        # data aug
        if self.mode == "train":
            transform = albumentations.Compose([
                albumentations.RandomRotate90(),
                albumentations.VerticalFlip(),
                albumentations.HorizontalFlip()
            ],
                is_check_shapes=False
            )
            transformed = transform(image=lr_data, mask=hr_data)
            lr_data, hr_data = transformed["image"], transformed["mask"]
        lr_data, hr_data = lr_data.transpose([2, 0, 1]), hr_data.transpose([2, 0, 1])
        return lr_data, hr_data, str(Path(hr_img_file).name)
