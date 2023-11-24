import argparse
import csv
import dataclasses
import logging
import os
import pathlib
import shutil
import typing

import plotnine as pn
import polars as pl
import tqdm

import infer as inferlib

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
parser.add_argument("data_dir", type=pathlib.Path)
parser.add_argument("work_dir", type=pathlib.Path)
parser.add_argument("--cuda", action="store_true", help="Enable CUDA (default: False)")
parser.add_argument("--checkpoint-path", type=pathlib.Path, help="Path to the checkpoint file")
parser.add_argument("--redo", action="store_true", default=False)
args = parser.parse_args()

data_dir: pathlib.Path = args.data_dir
work_dir: pathlib.Path = args.work_dir

if args.redo and work_dir.exists():
    shutil.rmtree(work_dir)
os.makedirs(work_dir, exist_ok=True)

if not args.cuda:
    device = "cpu"
else:
    device = "cuda:0"

model = inferlib.load_seg_model(str(args.checkpoint_path), device=device)

# results.csv
results_csv_path = work_dir / "results.csv"
segmentation_path = work_dir / "segmentation"
if not results_csv_path.exists():
    logger.info("beginning to process samples")
    with results_csv_path.open("w") as f:
        wrote_header = False
        other_fields = ["name"]
        fc = csv.writer(f)

        sample_dir_count = sum(1 for dir in data_dir.iterdir() if dir.is_dir())
        for sample_dir in tqdm.tqdm(data_dir.iterdir(), "running model on samples", sample_dir_count):
            if not sample_dir.is_dir():
                continue
            
            # logger.info("processing sample '%s'", sample_dir.name)
            result = inferlib.infer(
                model,
                sample_dir / "image.png",
                cuda=args.cuda,
                label_stuff=True,
                label_path=sample_dir / "label.png",
                output_dir=segmentation_path / sample_dir.name,
            )
            result = flatten_dict(dataclasses.asdict(result))

            if not wrote_header:
                wrote_header = True
                infer_fields = list(result.keys())
                fc.writerow(other_fields + infer_fields)

            fc.writerow([sample_dir.name] + [result[field] for field in infer_fields])
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
    )
    .vstack(
        df.select(
            "name",
            pl.col("cancer_metrics.iou").alias("iou"),
            pl.col("cancer_metrics.proportion").alias("proportion"),
            pl.lit("cancer").alias("class"),
        )
    )
)

(pn.ggplot(df, pn.aes("class", "iou"))
    + pn.geom_boxplot()
).save(work_dir / "box.png")

(pn.ggplot(df, pn.aes(x="iou"))
    + pn.xlim(0, 1)
    + pn.geom_histogram(bins=10)
    + pn.facet_wrap("class")
).save(work_dir / "hist.png")
