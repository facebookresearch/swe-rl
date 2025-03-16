# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import re
from dataclasses import dataclass, field
from typing import Literal, cast

from datasets import disable_caching, load_dataset
from transformers import HfArgumentParser


@dataclass(frozen=True)
class Args:
    output_file: str = field(metadata={"help": "The output file"})
    dataset: Literal[
        "princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"
    ] = field(
        default="princeton-nlp/SWE-bench_Verified",
        metadata={"help": "The dataset to be analyzed"},
    )
    num_proc: int = field(default=32, metadata={"help": "Number of processes to use"})


# Regex to match diff --git a/{file_name}.py b/...
def get_all_filenames(diff: str):
    return re.findall(r"diff --git a/(.*\.py) b/", diff)


def map_to_files(d: dict) -> dict:
    gt_files = get_all_filenames(d["patch"])
    if d["instance_id"] == "astropy__astropy-13398":
        # "New file"
        gt_files.remove(
            "astropy/coordinates/builtin_frames/itrs_observed_transforms.py"
        )
    return dict(
        instance_id=d["instance_id"],
        found_files=gt_files,
    )


if __name__ == "__main__":
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    dataset = load_dataset(args.dataset, split="test")

    disable_caching()
    dataset = dataset.map(
        map_to_files,
        remove_columns=dataset.column_names,
        num_proc=args.num_proc,
    )
    dataset.to_json(args.output_file, lines=True)
