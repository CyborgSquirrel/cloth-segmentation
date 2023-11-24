import argparse
import os
import pathlib
import shutil
import typing
import dataclasses
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from data.base_dataset import Normalize_image
from networks import U2NET

OUTPUT = pathlib.Path("./output")


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)


def load_seg_model(checkpoint_path, device="cpu"):
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


@dataclasses.dataclass
class Metrics:
    iou: typing.Optional[float] = None
    proportion: typing.Optional[float] = None


@dataclasses.dataclass
class InferResult:
    benign_metrics: typing.Optional[Metrics] = None
    cancer_metrics: typing.Optional[Metrics] = None


def compute_metrics(
    label: np.ndarray,
    output: np.ndarray,
) -> Metrics:
    metrics = Metrics()
    
    label_count = np.sum(label)
    output_count = np.sum(output)

    union = np.sum(np.logical_or(label, output))
    inter = np.sum(np.logical_and(label, output))
    if label_count == 0:
        iou = None
    else:
        iou = 0.0
        if union != 0:
            iou = inter / union
    metrics.iou = iou
    metrics.proportion = (
        output_count / (output.shape[1] * output.shape[2])
    )

    return metrics
    

def infer(
    checkpoint: pathlib.Path | U2NET,
    image_path: pathlib.Path,

    *,

    cuda: bool = False,

    label_stuff: bool = False,
    label_path: typing.Optional[pathlib.Path] = None,
    guess_label: bool = True,

    output_dir: typing.Optional[pathlib.Path] = None,
) -> InferResult:
    result = InferResult()
    
    if cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    if label_stuff:
        if label_path is None:
            if guess_label:
                label_path = image_path.absolute().parent / "label.png"
            else:
                raise ValueError("can't determine label_path")

        im_label = Image.open(label_path)
        im_label = im_label.resize((768, 768), Image.Resampling.NEAREST)
        im_label = np.asarray(im_label)
        im_label = im_label // 64

    if isinstance(checkpoint, pathlib.Path) or isinstance(checkpoint, str):
        model = load_seg_model(checkpoint, device=device)
    else:
        model = checkpoint
    im = Image.open(image_path).convert("RGB")

    im_size = im.size
    im = im.resize((768, 768), Image.BICUBIC)
    im_tensor = apply_transform(im)
    im_tensor = torch.unsqueeze(im_tensor, 0)

    if output_dir is not None:
        alpha_dir = output_dir / "alpha"
        seg_dir = output_dir / "seg"

        os.makedirs(alpha_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

    with torch.no_grad():
        output_tensor = model(im_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    classes_to_save = []

    # Check which classes are present in the image
    for cls in range(1, 4):  # Exclude background class (0)
        if np.any(output_arr == cls):
            classes_to_save.append(cls)

    # Save alpha masks
    for cls in classes_to_save:
        alpha_mask = (output_arr == cls).astype(np.uint8) * 255
        alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
        alpha_mask_img = Image.fromarray(alpha_mask, mode="L")
        alpha_mask_img = alpha_mask_img.resize(im_size, Image.BICUBIC)
        if output_dir is not None:
            alpha_mask_img.save(os.path.join(alpha_dir, f"{cls}.png"))

    if label_stuff:
        result.benign_metrics = compute_metrics(
            im_label == 2,
            output_arr == 2,
        )
        result.cancer_metrics = compute_metrics(
            im_label == 3,
            output_arr == 3,
        )
    
    # Save final cloth segmentations
    if output_dir is not None:
        cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8)*64, mode="L")
        cloth_seg = cloth_seg.resize(im_size, Image.BICUBIC)
        cloth_seg.save(seg_dir / "final_seg.png")

        shutil.copy(label_path, seg_dir / "label.png")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help to set arguments for Cloth Segmentation.")
    parser.add_argument("--image", type=pathlib.Path, help="Path to the input image")
    parser.add_argument("--label", type=pathlib.Path, help="Path to the label image", default=None)
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA (default: False)")
    parser.add_argument("--checkpoint-path", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()

    infer(
        args.checkpoint_path,
        args.image,
        cuda=args.cuda,
        label_stuff=True,
        label_path=args.label,
        output_dir=pathlib.Path("output"),
    )
