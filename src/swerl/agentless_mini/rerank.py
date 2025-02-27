# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""NOTE: this file is mostly not refactored."""

import argparse
import concurrent.futures
import json
import os
import re
from collections import Counter
from pathlib import Path

import swerl.agentless_mini.utils as utils

execution_results: dict[str, list[dict]] = {}


def _load_results(args):

    if os.path.exists(args.regression_test_file):
        print("Loading regression test file:", args.regression_test_file)
        data = utils.misc.load_jsonl(args.regression_test_file)
        considered_regression_tests = {
            d["instance_id"]: d["tests_passing_in_original_repo"] for d in data
        }
    else:
        considered_regression_tests = {}

    global execution_results

    roots = [Path(folder) for folder in args.patch_folder.split(",")]

    intervals = [(0, int(args.num_samples / len(roots)) - 1) for _ in range(len(roots))]

    interval = intervals[0]
    for i in range(interval[0], interval[1] + 1):
        for _, root in enumerate(roots):
            patches = utils.misc.load_jsonl(root / f"output_{i}_normalized.jsonl")
            print(
                f"Loaded {len(patches)} patches from {root / f'output_{i}_normalized.jsonl'}"
            )
            if "regression" in args.applied_rankers:
                try:
                    regression_test_results = utils.misc.load_jsonl(
                        root / f"output_{i}_regression_test_results.jsonl"
                    )
                except FileNotFoundError:
                    regression_test_results = []
            if "reproduction" in args.applied_rankers:
                try:
                    reproduction_test_results = utils.misc.load_jsonl(
                        root / f"output_{i}_reproduction_test_results.jsonl"
                    )
                except FileNotFoundError:
                    reproduction_test_results = []

            for patch in patches[:]:
                if "regression" in args.applied_rankers:
                    matching_results = [
                        x
                        for x in regression_test_results
                        if x["instance_id"] == patch["instance_id"]
                    ]
                    if (
                        len(matching_results) > 0
                        and "regression" in matching_results[0]
                    ):
                        tests_passing_in_original_repo = (
                            considered_regression_tests.get(patch["instance_id"], [])
                        )
                        failed_regression_tests = matching_results[0]["regression"]
                        if patch["instance_id"] in considered_regression_tests:
                            tests_passing_in_original_repo = (
                                considered_regression_tests[patch["instance_id"]]
                            )
                            # How many *real* regression tests failed
                            actual_failed_regression_tests = [
                                test
                                for test in failed_regression_tests
                                if test in tests_passing_in_original_repo
                            ]
                            # if failed_regression_tests != actual_failed_regression_tests and actual_failed_regression_tests != []:
                            #     breakpoint()
                            failed_regression_tests = actual_failed_regression_tests

                        # This quantity indicates how many regression tests failed
                        regression_test_result = len(failed_regression_tests)
                    else:
                        regression_test_result = 10000
                else:
                    regression_test_result = 0

                if "reproduction" in args.applied_rankers:
                    reproduction_test_data = [
                        x
                        for x in reproduction_test_results
                        if x["instance_id"] == patch["instance_id"]
                    ]
                    if args.max_tests > 0:
                        reproduction_test_data = reproduction_test_data[
                            : args.max_tests
                        ]

                    # if len(reproduction_test_data) > 0:
                    # NOTE: a hack now. use CodeT after the fix
                    # fixed (12/31/24)
                    resolved_tests = [
                        d["test_id"] for d in reproduction_test_data if d["resolved"]
                    ]
                    reproduction_test_result = frozenset(resolved_tests)
                else:
                    reproduction_test_result = frozenset()

                normalized_patch = patch["normalized_patch"].strip()
                model_patch = patch["model_patch"]
                execution_results.setdefault(patch["instance_id"], []).append(
                    {
                        "normalized_patch": normalized_patch,
                        "patch": model_patch,
                        "regression_test_result": regression_test_result,
                        "reproduction_test_result": reproduction_test_result,
                    }
                )


def get_sample(instance_id, sample_id) -> dict:
    """Returns the diff and pass status."""
    return execution_results[instance_id][sample_id]


def get_all_patches(instance_id, num_samples, deduplicate) -> list[tuple[int, str]]:
    """Returns all unique patches."""
    patches = [execution_results[instance_id][i]["patch"] for i in range(num_samples)]
    if deduplicate:
        patch_keys = [
            execution_results[instance_id][i]["normalized_patch"]
            for i in range(num_samples)
        ]
    else:
        patch_keys = [
            execution_results[instance_id][i]["patch"] for i in range(num_samples)
        ]
    unique_patches = set()
    patch_ids = []
    for i in range(num_samples):
        patch_key = patch_keys[i].strip()
        if patch_key and patch_key not in unique_patches:
            unique_patches.add(patch_key)
            patch_ids.append(i)
    return [(id, patches[id]) for id in patch_ids]


def modified_length(normalized_patch):
    changed_length = 0

    for line in normalized_patch.splitlines():
        if len(line) > 3 and (line.startswith("---") or line.startswith("+++")):
            continue

        if line.startswith("-"):
            changed_length += 1
        if line.startswith("+"):
            changed_length += 1

    assert changed_length != 0

    return changed_length


# Regex to match diff --git a/{file_name}.py b/...
def get_all_filenames(diff: str):
    return re.findall(r"diff --git a/(.*\.py) b/", diff)


def majority_voting(args):

    delta = 0
    with open(args.output_file, "w") as f:

        for instance_id in execution_results:
            if len(execution_results[instance_id]) < args.num_samples:
                print(
                    f"There were only {len(execution_results[instance_id])} patches for {instance_id} instead of the full {args.num_samples}"
                )

            patch_keys = [
                execution_results[instance_id][i]["normalized_patch"]
                for i in range(len(execution_results[instance_id]))
            ]
            regression_tests = [
                execution_results[instance_id][i]["regression_test_result"]
                for i in range(len(execution_results[instance_id]))
            ]

            # min_tests = min(regression_tests)
            # regression_tests = [
            #     True if x == min_tests else False for x in regression_tests
            # ]

            reproduction_tests = [
                execution_results[instance_id][i]["reproduction_test_result"]
                for i in range(len(execution_results[instance_id]))
            ]

            patch_ids_no_exec = [
                i
                for i in range(len(execution_results[instance_id]))
                if patch_keys[i].strip()
            ]

            def select_patch_ids_by_regression(patch_ids: list[int]) -> list[int]:
                if len(patch_ids) == 0:
                    return patch_ids

                min_failed_tests = min(
                    regression_tests[i] for i in patch_ids for i in patch_ids
                )
                return [i for i in patch_ids if regression_tests[i] == min_failed_tests]

            def select_patch_ids_by_reproduction(patch_ids: list[int]) -> list[int]:
                if len(patch_ids) == 0:
                    return patch_ids

                patch_ids_passing_same_tests = dict[frozenset[str], list[int]]()
                for patch_id in patch_ids:
                    patch_ids_passing_same_tests.setdefault(
                        reproduction_tests[patch_id], []
                    ).append(patch_id)
                consensus_set_status = "\n\n".join(
                    f"Num Tests: {len(k)}, Num Patches: {len(v)}"
                    for k, v in patch_ids_passing_same_tests.items()
                )
                # for k, v in patch_ids_passing_same_tests.items():
                #     assert len(v) == 1 or len(k) == 0
                print(f"Consensus set status:\n{consensus_set_status}")
                # Select the biggest consensus set by multiplying #tests and #patches
                assert len(patch_ids_passing_same_tests) > 0
                max_score = max(
                    len(k) * len(k) * len(v)
                    # len(k)
                    for k, v in patch_ids_passing_same_tests.items()
                )
                tests_keys = [
                    k
                    for k, v in patch_ids_passing_same_tests.items()
                    # if len(k) == max_score
                    if len(k) * len(k) * len(v) == max_score
                ]
                patch_ids = [
                    patch_id
                    for tests_key in tests_keys
                    for patch_id in patch_ids_passing_same_tests[tests_key]
                ]
                assert len(patch_ids) > 0
                print(f"Selected {len(patch_ids)} patches for {len(tests_keys)} tests.")
                return patch_ids

            patch_ids = patch_ids_no_exec
            regression_only = select_patch_ids_by_regression(patch_ids)
            reproduction_only = select_patch_ids_by_reproduction(patch_ids)
            regression_then_reproduction = select_patch_ids_by_reproduction(
                regression_only
            )
            reproduction_then_regression = select_patch_ids_by_regression(
                reproduction_only
            )
            considered = [
                regression_then_reproduction,
                reproduction_then_regression,
                reproduction_only,
                regression_only,
            ]
            considered = [
                ids for ids in considered if len(ids) > 0 and len(ids) != len(patch_ids)
            ]
            if len(considered) < 4:
                print("This is activated")
            if len(considered) == 0:
                patch_ids = []
            else:
                patch_ids = considered[0]

            if len(patch_ids) == 0:
                patch_ids = patch_ids_no_exec

            if len(patch_ids) == 0:
                print(f"No raw patches valid for {instance_id}")
                result = {
                    "model_name_or_path": "agentless",
                    "instance_id": instance_id,
                    "model_patch": "",
                }
                f.write(json.dumps(result) + "\n")
                continue

            def get_selected_id(patch_ids):
                vote = Counter()  # type: ignore
                first_appear_idx = dict()
                changed_length_idx = dict()
                for i in patch_ids:
                    sample = get_sample(instance_id, i)
                    patch_key = sample["normalized_patch"]
                    vote[patch_key] += 1
                    if patch_key not in first_appear_idx:
                        first_appear_idx[patch_key] = i
                        changed_length_idx[patch_key] = modified_length(patch_key)

                maj_selected_id = max(
                    patch_ids,
                    key=lambda i: (
                        vote[patch_keys[i]],
                        -first_appear_idx[patch_keys[i]],
                    ),
                )
                return maj_selected_id

            maj_selected_id = get_selected_id(patch_ids)
            maj_selected_id_no_exec = get_selected_id(patch_ids_no_exec)

            delta += int(maj_selected_id != maj_selected_id_no_exec)
            sample = get_sample(instance_id, maj_selected_id)

            result = {
                "model_name_or_path": "agentless",
                "instance_id": instance_id,
                "model_patch": sample["patch"],
            }

            f.write(json.dumps(result) + "\n")
    print(f"Delta: {delta}. (Total: {len(execution_results)})")


def normalize_one_patch(output_folder: Path, i: int):
    if os.path.exists(output_folder / f"output_{i}_normalized.jsonl"):
        return
    patches = utils.misc.load_jsonl(output_folder / f"output_{i}_processed.jsonl")
    for d in patches:
        instance_id = d["instance_id"]
        patch = d["model_patch"]
        original_file_content = d["original_file_content"]
        new_file_content = d["new_file_content"]
        edited_files = d["edited_files"]
        normalized_patch = utils.data.normalize_patch(
            instance_id,
            patch,
            original_file_content,
            new_file_content,
            edited_files,
        )
        d["normalized_patch"] = normalized_patch
    with open(output_folder / f"output_{i}_normalized.jsonl", "w") as f:
        for d in patches:
            f.write(json.dumps(d) + "\n")


def normalize_patches(args):
    # separate the patch folders
    output_folders = [Path(folder) for folder in args.patch_folder.split(",")]
    num_folders = len(output_folders)
    selected_ids = list(range(int(args.num_samples / num_folders)))

    for output_folder in output_folders:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(normalize_one_patch, output_folder, i)
                for i in selected_ids
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_folder", type=str)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--output_file", type=str, default="all_preds.jsonl")
    parser.add_argument(
        "--regression_test_file", type=str, default="__THIS_should_NEVER_exist_"
    )
    parser.add_argument(
        "--all_file_paths_file", type=str, default="__THIS_should_NEVER_exist_"
    )
    parser.add_argument(
        "--max_tests",
        type=int,
        default=-1,
        help="Max tests to consider. Useful for ablation.",
    )

    def ranker_type(value: str) -> str:
        value = value.lower()
        assert value in ["regression", "reproduction", "none"]
        return value

    parser.add_argument(
        "--applied_rankers",
        nargs="+",  # '+' means one or more arguments
        type=ranker_type,
        default=["none"],
        help='Apply the rankers in the provided order. Values should be in ["none", "regression", "reproduction"]',
    )

    args = parser.parse_args()
    print("Applying rankers: ", args.applied_rankers)

    # first normalize
    normalize_patches(args)
    # then load results
    _load_results(args)
    # then rerank
    majority_voting(args)


if __name__ == "__main__":
    main()
#
