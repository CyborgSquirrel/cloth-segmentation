"""
Extracts relevant slices from dicom files, and saves them as 16 bit png files.
"""

import argparse
import csv
import logging
import os
import os.path
import pathlib
import re
import shutil
import sys

import polars as pl
import pydicom as dicomlib
from PIL import Image

import duke_dbt_data

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=__file__[:-3] + ".log",
    level=logging.INFO,
)

# cli args
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=pathlib.Path)
parser.add_argument("dest_dir", type=pathlib.Path)
parser.add_argument("--force", "-f", action="store_true", default=False)
args = parser.parse_args()

# dest_dir
if args.dest_dir.exists():
    if args.force:
        shutil.rmtree(args.dest_dir)
    else:
        print("dest_dir already exists")
        exit(1)
os.makedirs(args.dest_dir)

data_dir = args.data_dir
boxes_path = data_dir / "boxes.csv"
file_paths_path = data_dir / "file-paths.csv"
manifest_path = data_dir / "manifest"

df_boxes = pl.read_csv(boxes_path)
df_file_paths = pl.read_csv(file_paths_path)
df = (
    df_boxes.lazy()
    .join(
        df_file_paths.lazy(),
        on=("PatientID", "StudyUID", "View"),
    )
    .groupby(
        ("descriptive_path", "PatientID", "StudyUID", "View")
    )
    .agg(
        pl.col("Slice").suffix("s")
    )
    .collect()
)

logging.info("beginning to process images")
for row in df.iter_rows(named=True):
    image_path = manifest_path / row["descriptive_path"]

    # correct messed up path part
    parts = list(image_path.parts)
    part = parts[-2]
    match_ = re.match("(\\d+.\\d+)-(\\d+)", part)
    parts[-2] = match_.group(1) + "-NA-" + match_.group(2)

    logging.info("processing image at '%s'", image_path)

    image_path = pathlib.Path(os.path.join(*parts))
    if not image_path.exists():
        logging.error("image doesn't exist")
        continue

    img = duke_dbt_data.dcmread_image(image_path, row["View"])

    for slice_index in row["Slices"]:
        slice_ar = img[slice_index]
        slice_img = Image.fromarray(slice_ar)
        slice_name = ".".join([row["PatientID"], row["StudyUID"], row["View"], str(slice_index)])
        slice_name += ".png"
        slice_img.save(args.dest_dir / slice_name)
    logging.info("successfully processed image")
