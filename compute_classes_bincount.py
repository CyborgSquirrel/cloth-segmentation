import argparse
import json
import pathlib

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=pathlib.Path)
parser.add_argument("output_file", type=pathlib.Path)
args = parser.parse_args()

data_dir: pathlib.Path = args.data_dir
output_file: pathlib.Path = args.output_file

total_count = np.zeros(4, int)

for sample_dir in data_dir.iterdir():
    if not sample_dir.is_dir():
        continue

    label_path = sample_dir / "label.png"
    im = Image.open(label_path)
    im = np.asarray(im) // 64
    im = im.flatten()

    im_count = np.bincount(im)
    im_count.resize(4)

    total_count += im_count

total_count = total_count.tolist()

with output_file.open("w") as f:
    json.dump(total_count, f)
