
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class AnimalDatasetParallel(Dataset):
    def __init__(self, data, directory, class_to_idx, transform):
        self.samples = [
            (Path(directory) / f"{image_id}.jpg", class_to_idx[label])
            for image_id, label in data[["image_id", "class"]].itertuples(index=False, name=None)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))

        return image, label
