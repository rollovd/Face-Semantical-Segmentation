from typing import Tuple, List, Any, Optional
from torch.utils.data import Dataset
import os
import albumentations as A
import numpy as np
import torch
import pickle
from torchvision.transforms import ToTensor
from PIL import Image

class FASSEG(Dataset):
    """
    Labels semantic:
    0: Background, 1: Hair, 2: Eyebrows, 3: Eyes, 4: Nose, 5: Mouth
    """
    def __init__(
        self,
        data_path: str,
        phase: str,
        augment: bool,
        img_size: Tuple[int, int],
        dataset_folder: str,
        weights_rgb_path: str
    ) -> None:

        self.data_path = data_path
        self.dataset_folder = dataset_folder
        self.phase = phase
        self.augment = augment
        self.img_size = img_size
        self.weights_rgb_path = weights_rgb_path

        if self.dataset_folder == 'V2':
            self.items = [filename.split('.')[0] for filename in os.listdir(f'{self.data_path}/{self.dataset_folder}/{self.phase}_RGB')]

        elif self.dataset_folder == 'V3':
            self.items, self.labels = self._get_files()

        self.kmeans = self._define_kmeans_mask()

        if self.augment:
            self.Resize = A.RandomScale()
            self.Crop = A.RandomCrop(width=self.img_size, height=self.img_size)
            # self.Rotate = A.Rotate(limit=3, p=0.5)
            self.HorizontalFlip = A.HorizontalFlip(p=1)
            self.RGBShift = A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.5
            )

            self.RandomBrightness = A.RandomBrightness(limit=0.2)

            self.transform = A.Compose(
                [
                 self.Resize,
                 self.Crop,
                #  self.Rotate,
                 self.HorizontalFlip,
                 self.RGBShift,
                #  self.RandomBrightness
                ]
            )

        else:
            self.Crop = A.RandomCrop(width=self.img_size, height=self.img_size)
            self.transform = A.Compose(
                [
                 self.Crop
                ]
            )
        
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.dataset_folder == 'V2':
            image = Image.open(f'{self.data_path}/{self.dataset_folder}/{self.phase}_RGB/{self.items[index]}.bmp')
            mask = Image.open(f'{self.data_path}/{self.dataset_folder}/{self.phase}_Labels/{self.items[index]}.bmp')

        elif self.dataset_folder == 'V3':
            image = Image.open(self.items[index])
            mask = Image.open(self.labels[index])

        if self.phase == 'Train':

            image = np.asarray(image)
            mask = np.asarray(mask)

            mask = self._discrete_mask(mask)
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        elif self.phase == 'Test':
            mask = mask.resize((self.img_size, self.img_size), Image.ANTIALIAS)
            mask = np.asarray(mask)
            mask = self._discrete_mask(mask)

            image = image.resize((self.img_size, self.img_size), Image.ANTIALIAS)
            image = np.asarray(image)

        image = self.to_tensor(image.copy())
        mask = torch.from_numpy(mask.copy()).long()

        return image, mask

    def _define_kmeans_mask(self):
        with open(self.weights_rgb_path, 'rb') as f:
            kmeans = pickle.load(f)

        return kmeans

    def _discrete_mask(self, mask: np.ndarray) -> np.ndarray:
        params = mask.shape
        reshape_mask = mask.reshape(params[0] * params[1], 3)
        new_mask = self.kmeans.predict(reshape_mask).reshape(params[0], params[1])

        return new_mask

    def _get_files(self):
        items = []
        labels = []

        path = f'{self.data_path}/{self.dataset_folder}/{self.phase}'
        path_rgb = path + '_RGB'
        path_labels = path + '_labels'

        for folder in os.listdir(path_labels):
            if folder != 'README.md':
                images = os.path.join(path_rgb, folder)
                for image in os.listdir(images):
                    items.append(os.path.join(images, image))

        for folder in os.listdir(path_labels):
            if folder != 'README.md':
                images = os.path.join(path_labels, folder)
                for image in os.listdir(images):
                    labels.append(os.path.join(images, image))

        return sorted(items), sorted(labels)
