"""
The script reads:

- slices extracted from dicom files
- boxes.csv file with the lesions present in the slices

Then, for each of the slices, the script:

- computes a new image where each pixel represents a label for the pixel at the
  same coordinate in the original image, using the boxes.csv file and image
  processing techniques
- crops out the background of the original image and the labels image
- saves the resulting images
"""

import argparse
import dataclasses
import logging
import os
import pathlib
import re
import shutil
import typing

import numpy as np
import polars as pl
from PIL import Image, ImageChops, ImageDraw, ImageOps

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=__file__[:-3] + ".log",
    level=logging.INFO,
)

# EXAMPLE: DBT-P00303.DBT-S02436.rcc.21.png
RE_NAME = re.compile(
    "(?P<patient_id>[^.]*)" "\\."
    "(?P<study_uid>[^.]*)" "\\."
    "(?P<view>[^.]*)" "\\."
    "(?P<slice>[^.]*)"
)


@dataclasses.dataclass
class Lesion:
    x: int
    y: int
    width: int
    height: int
    class_: str

    def from_dict(lesion: dict):
        return Lesion(
            x=lesion["X"],
            y=lesion["Y"],
            width=lesion["Width"],
            height=lesion["Height"],
            class_=lesion["Class"],
        )


def compute_background_mask(im: Image):
    # threshold image
    im = np.asarray(im)
    im = (im > 0).astype(np.uint8) * 64
    im = Image.fromarray(im).copy()  # doesn't work if I don't copy here

    # flood fill background
    # NOTE: Since all the images are pointing right, we start the flood fill in
    # the upper right corner.
    ImageDraw.floodfill(im, (im.size[0]-1, 0), 128)

    # compute final mask
    im = np.asarray(im)
    im = im != 128
    im = Image.fromarray(im)

    return im


def process_image(im_path: str, lesions: list[Lesion]) -> tuple[Image, Image]:
    im_path = pathlib.Path(im_path)
    im = Image.open(im_path)

    match_ = re.match(RE_NAME, im_path.name)
    laterality = match_.group("view")[0].upper()

    # make sure all images are pointing right
    if laterality == "R":
        im = ImageOps.mirror(im)
    
    im_background = compute_background_mask(im)

    im_label = Image.new("L", im.size, 0)
    im_label = Image.composite(Image.new("L", im.size, 64*1), im_label, im_background)

    # add lesions
    for lesion in lesions:
        # mirror the lesions if we mirrored the image
        x = lesion.x
        if laterality == "R":
            x = im.size[0] - x - lesion.width
        
        im_lesion = Image.new("1", im.size)
        draw = ImageDraw.Draw(im_lesion)
        draw.rectangle(
            (
                x                       ,
                lesion.y                ,
                x        + lesion.width ,
                lesion.y + lesion.height,
            ),
            1,
        )
        im_lesion = ImageChops.logical_and(im_lesion, im_background)

        if lesion.class_ == "benign":
            color = 64*2
        elif lesion.class_ == "cancer":
            color = 64*3
        else:
            raise ValueError("invalid lesion class")
        
        im_label = Image.composite(
            Image.new("L", im.size, color),
            im_label,
            im_lesion,
        )

    # crop images
    bbox = im_background.getbbox()
    im = im.crop(bbox)
    im_label = im_label.crop(bbox)

    return (im, im_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=pathlib.Path)
    parser.add_argument("boxes_path", type=pathlib.Path)
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

    dest_dir_image = args.dest_dir / "image"
    os.makedirs(dest_dir_image)

    dest_dir_label = args.dest_dir / "label"
    os.makedirs(dest_dir_label)

    # boxes.csv
    df_boxes = pl.read_csv(args.boxes_path)
    df_boxes = (
        df_boxes
        .with_columns(
            pl.struct("X", "Y", "Width", "Height", "Class").alias("Lesion")
        )
        .groupby("PatientID", "StudyUID", "View", "Slice")
        .agg(pl.col("Lesion").suffix("s"))
        .with_columns(
            (
                  pl.col("PatientID") + pl.lit(".")
                + pl.col("StudyUID") + pl.lit(".")
                + pl.col("View").str.to_lowercase() + pl.lit(".")
                + pl.col("Slice").cast(pl.Utf8) + pl.lit(".png")
            ).alias("FileName")
        )
    )

    for row in df_boxes.iter_rows(named=True):
        lesions = [Lesion.from_dict(lesion) for lesion in row["Lesions"]]
        image_path = args.data_dir / row["FileName"]
        logging.info("processing image at '%s'", image_path)
        im, im_label = process_image(image_path, lesions)

        im.save(dest_dir_image / row["FileName"])
        im_label.save(dest_dir_label / row["FileName"])

        logging.info("successfully processed image")

