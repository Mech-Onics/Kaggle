
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset

class VariableAnimalDataset(Dataset):
    def __init__(self, data, directory, class_to_idx, max_side=256):
        self.samples = [
            (Path(directory) / f"{image_id}.jpg", class_to_idx[label])
            for image_id, label in data[["image_id", "class"]].itertuples(index=False, name=None)
        ]
        self.max_side = max_side

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((self.max_side, self.max_side), Image.Resampling.BILINEAR)
            image = to_tensor(img)

        return image, label


def variable_collate(batch):
    images, labels = zip(*batch)

    # Batch dimensions follow the largest image, minimum 8px for 3 pooling layers
    max_h = max(8, max(img.shape[1] for img in images))
    max_w = max(8, max(img.shape[2] for img in images))

    padded = []
    for img in images:
        h, w = img.shape[1:]
        left = (max_w - w) // 2
        top = (max_h - h) // 2
        padded.append(F.pad(
            img, (left, max_w-w-left, top, max_h-h-top),
            value=114/255
        ))

    return torch.stack(padded), torch.tensor(labels)
