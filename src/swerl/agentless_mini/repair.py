# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import asyncio
import concurrent.futures
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from datasets import Dataset
from tqdm.auto import tqdm

import swerl.agentless_mini.utils as utils


@dataclass(frozen=True)
class Args:
    # Localization file and output file for repair
    loc_file: str
    output_folder: str
    max_input_tokens: int = field(default=60000)

    @property
    def output_file(self):
        return (Path(self.output_folder) / "output.jsonl").as_posix()


def _post_process_multifile_repair(
    raw_output: str, file_contents: dict[str, str]
) -> tuple[list[str], list[str]]:
    edit_multifile_commands = utils.data.extract_python_blocks(raw_output)
    edited_files = list[str]()
    new_contents = list[str]()
    file_to_commands = utils.data.split_edit_multifile_commands(edit_multifile_commands)

    for edited_file_key in file_to_commands:
        edited_file = ""
        new_content = ""
        edit_commands = file_to_commands[edited_file_key]
        edited_file = edited_file_key
        if edited_file not in file_contents:
            continue

        content = file_contents[edited_file]
        new_content = utils.data.parse_diff_edit_commands(edit_commands, content)

        if edited_file == "" or new_content == "":
            continue
        edited_files.append(edited_file)
        new_contents.append(new_content)

    return edited_files, new_contents


TOKEN_COUNT_MAP: dict[tuple[str, str], int] = {}


def count_tokens(instance_id: str, file_name: str, content: str) -> int:
    key = (instance_id, file_name)
    if key in TOKEN_COUNT_MAP:
        return TOKEN_COUNT_MAP[key]
    tokens = utils.misc.count_tokens(content)
    TOKEN_COUNT_MAP[key] = tokens
    return tokens


def construct_topn_file_context(
    instance_id: str,
    pred_files: list[str],
    file_contents: dict[str, str],
    max_input_tokens: int,
    # Randomize the order of the contents
    randomize: bool = False,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    num_tokens = 0
    all_contents = list[str]()
    for pred_file in pred_files:
        content = file_contents[pred_file]
        content = f"### {pred_file}\n{content}"
        num_new_tokens = count_tokens(instance_id, pred_file, content)
        if num_tokens + num_new_tokens > max_input_tokens:
            continue
        num_tokens += num_new_tokens
        all_contents.append(content)

    if len(all_contents) == 0 and len(pred_files) > 0:
        return f"### {pred_files[0]}\n{file_contents[pred_files[0]]}"
    if randomize:
        random.shuffle(all_contents)
    return "\n\n".join(all_contents)


T = TypeVar("T")


async def process_loc(
    args: Args,
    inf_args: utils.args.InferenceArgs,
    # bench_args: utils.args.BenchArgs,
    client: utils.api.OpenAIClient,
    semaphore: asyncio.Semaphore,
    loc: dict,
    swe_bench_data: list[dict],
    prev_o: list[dict],
):
    instance_id = loc["instance_id"]
    found = any(o["instance_id"] == instance_id for o in prev_o)

    if found:
        print(f"skipping {instance_id} since patch already generated")
        return None

    # Backward compatibility
    if len(loc["found_files"]) == 0 or isinstance(loc["found_files"][0], str):
        loc["found_files"] = [loc["found_files"]]  # convert to list of list

    if all(len(x) == 0 for x in loc["found_files"]):
        print(f"no files found for {instance_id}")
        return None

    all_found_files: list[list[str]] = loc["found_files"]
    all_found_files = [
        pred_files for pred_files in all_found_files if len(pred_files) > 0
    ]
    # Add remaining found files from the first found few files
    assert len(all_found_files) > 0

    # only keep unique pred_files in all_found_files. all_found_files is a list[list[str]]
    unique_files_set = set[tuple[str, ...]]()
    unique_all_found_files: list[list[str]] = []
    for pred_files in all_found_files:
        # Convert the list to a tuple to make it hashable for the set
        pred_files_tuple = tuple(pred_files)
        if pred_files_tuple not in unique_files_set:
            unique_files_set.add(pred_files_tuple)
            unique_all_found_files.append(pred_files)
    all_found_files = unique_all_found_files[: inf_args.num_samples]

    assert len(all_found_files) > 0
    for index in range(inf_args.num_samples - len(all_found_files)):
        all_found_files.append(all_found_files[index % len(all_found_files)])
    assert len(all_found_files) == inf_args.num_samples

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    structure = utils.data.get_repo_structure(instance_id)
    repo_file_contents, _, _ = utils.data.get_full_file_paths_and_classes_and_functions(
        structure
    )
    repo_file_contents_dict = {path: lines for path, lines in repo_file_contents}

    def get_input_messages(context: str, system: str | None = None) -> list[dict]:
        content = utils.prompts.REPAIR.format(
            problem_statement=problem_statement,
            content=context,
        ).strip()
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": content})
        return messages

    # Construct file contents
    def _get_file_contents(pred_files: list[str]) -> dict[str, str]:
        return {
            pred_file: "\n".join(repo_file_contents_dict[pred_file])
            for pred_file in pred_files
            # # This should be always true except for one special GT case:
            # # astropy/coordinates/builtin_frames/itrs_observed_transforms.py
            # # This is fixed in the GT file (12/26/24).
            # if pred_file in repo_file_contents_dict
        }

    all_topn_contents = list[str]()
    randomize = inf_args.num_samples > 1
    for pred_files in all_found_files:
        # pred_files = pred_files[: args.top_n]
        # Construct file contents
        topn_content = construct_topn_file_context(
            instance_id,
            pred_files,
            _get_file_contents(pred_files),
            args.max_input_tokens,
            randomize=randomize,
        )
        all_topn_contents.append(topn_content)

    all_requests = [
        dict(
            model=inf_args.model,
            messages=get_input_messages(prompt),
            max_tokens=inf_args.max_tokens,
            temperature=inf_args.temperature,
            n=1,
        )
        for prompt in all_topn_contents
    ]
    del all_topn_contents

    idx_and_responses = await utils.api.collect_responses_async(
        client, semaphore, all_requests
    )
    assert len(idx_and_responses) == inf_args.num_samples
    indices = [idx for idx, _ in idx_and_responses]
    assert sorted(indices) == list(range(inf_args.num_samples))

    all_generations = list[str]()
    all_outputs = list[str]()
    all_trajs = list[dict]()
    all_prev_contents = list[list[str]]()
    all_file_names = list[list[str]]()
    for idx, response in idx_and_responses:
        request = all_requests[idx]
        file_contents = _get_file_contents(all_found_files[idx])
        prompt = request["messages"][-1]["content"]
        if response is not None:
            output = response.choices[0].message.content
        else:
            output = ""
        all_trajs.append(dict(prompt=prompt, response=output))

        all_generations.append(output)

        # Extract the <solution> part
        output = utils.api.parse_thinking_output(output)

        edited_files, new_contents = _post_process_multifile_repair(
            output, file_contents
        )

        if len(new_contents) == 0:
            all_prev_contents.append([])
            all_file_names.append([])
        else:
            prev_content = [file_contents[edited_file] for edited_file in edited_files]
            all_prev_contents.append(prev_content)
            all_file_names.append(edited_files)
        all_outputs.append(output)
    return dict(
        instance_id=instance_id,
        raw_output=all_outputs,
        all_generations=[all_generations],
        traj=all_trajs,
        prev_content=[all_prev_contents],
        file_names=[all_file_names],
        all_indices=indices,
        all_found_files=all_found_files,
    )


async def repair(
    args: Args,
    inf_args: utils.args.InferenceArgs,
    swe_bench_data: Dataset,
):
    locs = utils.misc.load_jsonl(args.loc_file)
    prev_o = (
        utils.misc.load_jsonl(args.output_file)
        if os.path.exists(args.output_file)
        else []
    )

    all_instance_ids = set(swe_bench_data["instance_id"])
    locs = [loc for loc in locs if loc["instance_id"] in all_instance_ids]

    client = utils.api.OpenAIClient()

    semaphore = asyncio.Semaphore(inf_args.max_concurrent_requests)
    all_tasks = [
        process_loc(args, inf_args, client, semaphore, loc, swe_bench_data, prev_o)
        for loc in locs
    ]
    pbar = tqdm(total=len(all_tasks), desc="Process all instances")
    for completion in asyncio.as_completed(all_tasks):
        result = await completion
        pbar.update(1)
        if result is not None:
            with open(args.output_file, "a") as f:
                f.write(json.dumps(result) + "\n")


def post_process_raw_output(raw_output_text: str, file_contents: dict[str, str]):
    git_diffs = ""
    raw_git_diffs = ""
    edited_files = list[str]()
    new_contents = list[str]()
    contents = list[str]()
    edited_files, new_contents = _post_process_multifile_repair(
        raw_output_text, file_contents
    )

    contents = [file_contents[edited_file] for edited_file in edited_files]

    git_diff = utils.data.fake_git_repo(
        utils.envs.PLAYGROUND_DIR, edited_files, contents, new_contents
    )

    raw_git_diffs += "\n" + git_diff.replace("\\ No newline at end of file\n", "")

    syntax_success = utils.data.check_syntax(new_contents)

    differ_by_empty_lines = utils.data.check_code_differ_by_just_empty_lines(
        new_contents, contents
    )

    if syntax_success and not differ_by_empty_lines:
        git_diffs = raw_git_diffs
    else:
        git_diffs = ""  # no need to evaluate

    return git_diffs, raw_git_diffs, contents, edited_files, new_contents


def post_process_repair(args: Args, select_id: int):
    """
    apply some diff formatting.
    """
    output_file = args.output_file.replace(".jsonl", f"_{select_id}_processed.jsonl")
    if os.path.exists(output_file):
        print(f"output file {output_file} already exists. skipping.")
        return

    raw_outputs = utils.misc.load_jsonl(args.output_file)
    data_to_write = list[dict]()
    for raw_output in raw_outputs:
        instance_id = raw_output["instance_id"]
        if raw_output["raw_output"] == "":
            data_to_write.append(
                dict(
                    model_name_or_path="agentless",
                    instance_id=instance_id,
                    model_patch="",
                )
            )
            continue

        assert select_id >= 0
        # Use the indexed generation
        generation_idx = select_id

        raw_output_text = raw_output["all_generations"][0][generation_idx]
        original_file_content = raw_output["prev_content"][0][generation_idx]
        pred_file = raw_output["file_names"][0][generation_idx]

        git_diffs = ""
        raw_git_diffs = ""
        if isinstance(raw_output["raw_output"], str):
            # for backward compatibility
            raw_output["raw_output"] = [raw_output["raw_output"]]

        if isinstance(original_file_content, str):
            original_file_content = [original_file_content]
            pred_file = [pred_file]

        file_contents = {
            file_name: o_file_content
            for file_name, o_file_content in zip(pred_file, original_file_content)
        }

        if raw_output_text:
            (
                git_diffs,
                raw_git_diffs,
                content,
                edited_files,
                new_contents,
            ) = post_process_raw_output(raw_output_text, file_contents)
        else:
            git_diffs = ""
            raw_git_diffs = ""
            content = []
            edited_files = []
            new_contents = []

        data_to_write.append(
            {
                "model_name_or_path": "agentless",
                "instance_id": instance_id,
                "model_patch": git_diffs.lstrip(),
                "raw_model_patch": raw_git_diffs.lstrip(),
                "original_file_content": content,
                "edited_files": edited_files,
                "new_file_content": new_contents,
            }
        )
    with open(output_file, "w") as f:
        for entry in data_to_write:
            f.write(json.dumps(entry) + "\n")


async def main(
    bench_args: utils.args.BenchArgs,
    inference_args: utils.args.InferenceArgs,
    args: Args,
):
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    meta = dict(
        bench_args=str(bench_args),
        inference_args=str(inference_args),
        args=str(args),
    )
    with (output_folder / "args.json").open("w") as f:
        json.dump(meta, f, indent=4)

    swe_bench_data = bench_args.load()
    await repair(args, inference_args, swe_bench_data)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(post_process_repair, args, i)
            for i in range(inference_args.num_samples)
        ]
        for future in concurrent.futures.as_completed(futures):
            # make sure to catch any exceptions here
            future.result()


if __name__ == "__main__":
    params = utils.args.parse_args_into_dataclasses(
        utils.args.BenchArgs,
        utils.args.InferenceArgs,
        Args,
    )
    asyncio.run(main(*params))
