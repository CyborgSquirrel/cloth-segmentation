import argparse
import dataclasses
import math
import os
import pathlib
import typing

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import tomosynthesis_dataset as dataset
from networks import U2NET
from options.base_options import Options

opt = Options()

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
        output_count / math.prod(output.shape)
    )

    return metrics
    

def infer(
    checkpoint: pathlib.Path | U2NET,
    image: str | pathlib.Path | torch.Tensor,

    *,

    cuda: bool = False,

    label_stuff: bool = False,
    label: typing.Optional[pathlib.Path | torch.Tensor] = None,
    guess_label: bool = True,

    output_dir: typing.Optional[pathlib.Path] = None,
) -> InferResult:
    if isinstance(image, str):
        image = pathlib.Path(image)
    if isinstance(label, str):
        label = pathlib.Path(label)
    
    result = InferResult()
    
    if cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    # label
    if label_stuff:
        if label is None:
            if guess_label and isinstance(image, pathlib.Path):
                label = image.absolute().parent / "label.png"
            else:
                raise ValueError("can't determine label_path")

        if isinstance(label, pathlib.Path):
            im_label = Image.open(label)
            im_label = dataset.transform_label(opt, im_label)
        else:
            im_label = label
        im_label = im_label.numpy()

    # model
    if isinstance(checkpoint, pathlib.Path) or isinstance(checkpoint, str):
        model = load_seg_model(checkpoint, device=device)
    else:
        model = checkpoint

    # image
    if isinstance(image, pathlib.Path):
        im = Image.open(image).convert("RGB")
        im_size = im.size
        im_tensor = dataset.transform(opt)(im)
    else:
        im_tensor = image
        im_size = tuple(im_tensor.size())[1:]
    im_tensor = im_tensor.unsqueeze(0)

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
        cloth_seg = cloth_seg.resize(im_size, Image.Resampling.NEAREST)
        cloth_seg.save(seg_dir / "final_seg.png")

        im_label = Image.fromarray(im_label.astype(np.uint8)*64, mode="L")
        im_label = im_label.resize(im_size, Image.Resampling.NEAREST)
        im_label.save(seg_dir / "label.png")

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
        label=args.label,
        output_dir=pathlib.Path("output"),
    )
