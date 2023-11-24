import collections
import pathlib

import torchvision
import torchvision.transforms as transforms
from PIL import Image

from data.base_dataset import BaseDataset, Normalize_image


class TomosynthesisDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.data_dir = pathlib.Path(opt.image_folder)
        self.width = opt.fine_width
        self.height = opt.fine_height
        
        self.image_info = collections.defaultdict(dict)

        for sample_dir in self.data_dir.iterdir():
            index = len(self.image_info)
            self.image_info[index]["image_path"] = sample_dir / "image.png"
            self.image_info[index]["label_path"] = sample_dir / "label.png"

        # for rgb imgs
        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [Normalize_image(opt.mean, opt.std)]
        self.transform_rgb = transforms.Compose(transforms_list)

        self.dataset_size = len(self.image_info)

    def __getitem__(self, index):
        info = self.image_info[index]
        
        im = Image.open(info["image_path"]).convert("RGB")
        im = im.resize((self.width, self.height), resample=Image.Resampling.BICUBIC)
        im = self.transform_rgb(im)

        im_label = Image.open(info["label_path"])
        im_label = im_label.resize((self.width, self.height), resample=Image.Resampling.NEAREST)
        im_label = torchvision.transforms.functional.pil_to_tensor(im_label)
        im_label = im_label.squeeze(0)
        im_label = im_label // 64

        return im, im_label
    
    def __len__(self):
        return len(self.image_info)

    def name(self):
        return "TomosynthesisDataset"
