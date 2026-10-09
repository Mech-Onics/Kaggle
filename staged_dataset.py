from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

FEATURES = {"Eye": 0, "Face": 1, "Mouth": 2, "Nose": 3}

class StagedImages(Dataset):
    def __init__(self, data, directory, mode="pad"):
        self.samples = []
        for image_id, label in data[["image_id", "class"]].itertuples(index=False, name=None):
            species = 0 if label.startswith("Cat") else 1
            feature = FEATURES[label[3:]]
            self.samples.append((Path(directory) / f"{image_id}.jpg", species, feature))
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, species, feature = self.samples[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.mode == "pad":
                img = ImageOps.pad(img, (128, 128), method=Image.Resampling.BILINEAR,
                                   color=(114, 114, 114))
            else:  # Previous stretched baseline, if its checkpoint exists
                img = img.resize((128, 128), Image.Resampling.BILINEAR)
            x = to_tensor(img)
        return x, species, feature, species * 4 + feature
