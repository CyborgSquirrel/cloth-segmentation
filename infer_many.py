import argparse
import csv
import dataclasses
import itertools
import logging
import os
import pathlib
import shutil
import typing

import plotnine as pn
import polars as pl
import torch
import torchvision
import tqdm

import infer as inferlib
from data.tomosynthesis_dataset import TomosynthesisDataset
from options.base_options import Options

opt = Options()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flatten_dict(d: dict[str, typing.Any]) -> dict[str, typing.Any]:
    result = dict()
    for key, value in d.items():
        if isinstance(value, dict):
            value = flatten_dict(value)
            for value_key, value_value in value.items():
                result[key + "." + value_key] = value_value
        else:
            result[key] = value

    return result


parser = argparse.ArgumentParser()
parser.add_argument("work_dir", type=pathlib.Path)
parser.add_argument("--cuda", action="store_true", help="Enable CUDA (default: False)")
parser.add_argument("--checkpoint-path", type=pathlib.Path, help="Path to the checkpoint file")
parser.add_argument("--redo", action="store_true", default=False)
args = parser.parse_args()

work_dir: pathlib.Path = args.work_dir

if args.redo and work_dir.exists():
    shutil.rmtree(work_dir)
os.makedirs(work_dir, exist_ok=True)

if not args.cuda:
    device = "cpu"
else:
    device = "cuda:0"

model = inferlib.load_seg_model(str(args.checkpoint_path), device=device)

# dataset
gen = torch.Generator()
gen.manual_seed(opt.seed)
dataset = TomosynthesisDataset(opt)
dataset.load_image = False
[dataset_train, dataset_val] = torch.utils.data.random_split(
    dataset,
    [1-opt.val_proportion, opt.val_proportion],
    generator=gen,
)

dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=1,
    pin_memory=True,  # TODO: only set to true if training on the gpu?
)
dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=1,
    pin_memory=True,  # TODO: only set to true if training on the gpu?
)

dataloader_iter = itertools.chain(
    zip(dataloader_train, itertools.repeat("train")),
    zip(dataloader_val, itertools.repeat("val")),
)

# results.csv
results_csv_path = work_dir / "results.csv"
segmentation_path = work_dir / "segmentation"
if not results_csv_path.exists():
    logger.info("beginning to process samples")
    with results_csv_path.open("w") as f:
        wrote_header = False
        other_fields = ["name", "kind"]
        fc = csv.writer(f)

        for data, kind in tqdm.tqdm(dataloader_iter, "running model on samples", len(dataset)):
            # logger.info("processing sample '%s'", sample_dir.name)
            result = inferlib.infer(
                model,
                image=data["image_path"][0],
                cuda=args.cuda,
                label_stuff=True,
                label=data["label_path"][0],
                output_dir=segmentation_path / data["name"][0],
            )
            result = flatten_dict(dataclasses.asdict(result))

            if not wrote_header:
                wrote_header = True
                infer_fields = list(result.keys())
                fc.writerow(other_fields + infer_fields)

            fc.writerow([data["name"], kind] + [result[field] for field in infer_fields])
else:
    logger.info("using saved results.csv")

# plots
df = pl.read_csv(results_csv_path)
df = (
    df.select(
        "name",
        pl.col("benign_metrics.iou").alias("iou"),
        pl.col("benign_metrics.proportion").alias("proportion"),
        pl.lit("benign").alias("class"),
        pl.col("kind"),
    )
    .vstack(
        df.select(
            "name",
            pl.col("cancer_metrics.iou").alias("iou"),
            pl.col("cancer_metrics.proportion").alias("proportion"),
            pl.lit("cancer").alias("class"),
            pl.col("kind"),
        )
    )
)

(pn.ggplot(df, pn.aes("class", "iou"))
    + pn.geom_boxplot()
    + pn.facet_wrap("kind")
).save(work_dir / "box.png")

(pn.ggplot(df, pn.aes(x="iou"))
    + pn.xlim(0, 1)
    + pn.geom_histogram(bins=10)
    + pn.facet_wrap(("kind", "class"))
).save(work_dir / "hist.png")
