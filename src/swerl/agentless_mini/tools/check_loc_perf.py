# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""Compute the localization performance"""

import functools
import json
import re
from dataclasses import dataclass, field
from typing import Literal, cast

from datasets import disable_caching, load_dataset
from transformers import HfArgumentParser


@dataclass(frozen=True)
class Args:
    locfile: str = field(metadata={"help": "The location file to be analyzed"})
    dataset: Literal[
        "princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"
    ] = field(
        default="princeton-nlp/SWE-bench_Verified",
        metadata={"help": "The dataset to be analyzed"},
    )
    num_proc: int = field(default=32, metadata={"help": "Number of processes to use"})
    max_samples: int = field(default=10000000000)


# Regex to match diff --git a/{file_name}.py b/...
def get_all_filenames(diff: str):
    return re.findall(r"diff --git a/(.*\.py) b/", diff)


def map_to_analysis(
    d_found_files: dict[str, list[str] | list[list[str]]], d: dict
) -> dict:
    instance_id = d["instance_id"]
    all_found_files = d_found_files[instance_id]
    if len(all_found_files) == 0 or isinstance(all_found_files[0], str):
        all_found_files = [all_found_files]  # type: ignore
    gt_files = get_all_filenames(d["patch"])
    all_coverages = [
        len(set(found_files).intersection(set(gt_files))) / len(set(gt_files))
        for found_files in all_found_files
    ]
    return dict(
        max_coverage=max(all_coverages),
        avg_coverage=sum(all_coverages) / len(all_coverages),
        all_files_found=any(
            set(gt_files).issubset(set(found_files)) for found_files in all_found_files
        ),
        any_file_found=any(
            len(set(found_files).intersection(set(gt_files))) > 0
            for found_files in all_found_files
        ),
        no_found_files=all(len(found_files) == 0 for found_files in all_found_files),
    )


def reduce_fn(x, y):
    return dict(
        all_files_found=int(x["all_files_found"]) + int(y["all_files_found"]),
        any_file_found=int(x["any_file_found"]) + int(y["any_file_found"]),
        no_found_files=int(x["no_found_files"]) + int(y["no_found_files"]),
        max_coverage=x["max_coverage"] + y["max_coverage"],
        avg_coverage=x["avg_coverage"] + y["avg_coverage"],
    )


if __name__ == "__main__":
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    dataset = load_dataset(args.dataset, split="test")
    loc_results = load_dataset("json", data_files=args.locfile)["train"]
    found_files = {
        d["instance_id"]: d["found_files"][: args.max_samples] for d in loc_results
    }
    dataset = dataset.filter(lambda x: x["instance_id"] in found_files)
    dataset = dataset.map(
        lambda x: map_to_analysis(found_files, x),
        remove_columns=dataset.column_names,
        num_proc=args.num_proc,
    )
    result = functools.reduce(reduce_fn, dataset)
    # It's the average of max coverage for each instance
    result["max_coverage"] /= len(dataset)
    result["avg_coverage"] /= len(dataset)
    result["total"] = len(dataset)
    print(json.dumps(result, indent=2))
