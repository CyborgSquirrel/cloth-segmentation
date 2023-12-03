import collections
import logging
import pathlib

import torchvision
import torchvision.transforms as transforms
from PIL import Image

from data.base_dataset import BaseDataset, Normalize_image

logger = logging.getLogger(__name__)


def transform(opt):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (opt.fine_width, opt.fine_height),
            transforms.InterpolationMode.BICUBIC,
        ),
        Normalize_image(opt.mean, opt.std),
    ])


def transform_label(opt, label: Image):
    label = label.resize(
        (opt.fine_width, opt.fine_height),
        resample=Image.Resampling.NEAREST,
    )
    label = torchvision.transforms.functional.pil_to_tensor(label)
    label = label.squeeze(0)
    label = label // 64
    return label


class TomosynthesisDataset(BaseDataset):
    def __init__(self, opt):
        self.load_image = False
        
        self.opt = opt
        self.data_dir = pathlib.Path(opt.image_folder)
        self.width = opt.fine_width
        self.height = opt.fine_height
        
        self.image_info = collections.defaultdict(dict)

        self.transform = transform(opt)

        for sample_dir in self.data_dir.iterdir():
            index = len(self.image_info)
            self.image_info[index]["image_path"] = sample_dir / "image.png"
            self.image_info[index]["label_path"] = sample_dir / "label.png"
            self.image_info[index]["name"] = sample_dir.name

        logger.info("created dataset with %s images", len(self.image_info))

    def __getitem__(self, index):
        info = self.image_info[index]
        result = dict()

        result["image_path"] = str(info["image_path"])
        if self.load_image:
            im = Image.open(info["image_path"]).convert("RGB")
            im = self.transform(im)
            result["image"] = im

        result["label_path"] = str(info["label_path"])
        if self.load_image:
            im_label = Image.open(info["label_path"])
            im_label = transform_label(self.opt, im_label)
            result["label"] = im_label

        result["name"] = info["name"]
        
        return result
    
    def __len__(self):
        return len(self.image_info)

    def name(self):
        return "TomosynthesisDataset"
