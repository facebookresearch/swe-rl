# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

import swerl.agentless_mini.utils as utils


@dataclass(frozen=True)
class Args:
    output_file: str


async def localize_instance(
    client: utils.api.OpenAIClient,
    semaphore: asyncio.Semaphore,
    bug: dict,
    args: utils.args.InferenceArgs,
    existing_instance_ids,
) -> dict | None:
    if bug["instance_id"] in existing_instance_ids:
        print(f"Skipping existing instance_id: {bug['instance_id']}")
        return None

    d = utils.data.get_repo_detail(bug["instance_id"])
    problem_statement = bug["problem_statement"]
    structure = d["structure"]

    # filter out None Python files
    utils.data.filter_none_python(structure)
    # filter out test files (unless its pytest)
    if not d["instance_id"].startswith("pytest"):
        utils.data.filter_out_test_files(structure)

    # file level localization
    found_files, file_traj = await localize_files(
        client,
        semaphore,
        problem_statement,
        structure,
        args,
    )

    return {
        "instance_id": d["instance_id"],
        "found_files": found_files,
        "file_traj": file_traj,
    }


async def localize_files(
    client: utils.api.OpenAIClient,
    semaphore: asyncio.Semaphore,
    problem_statement: str,
    structure: dict,
    args: utils.args.InferenceArgs,
):
    assert args.num_samples > 0

    found_files: list[str] = []

    def get_message():
        randomize = args.num_samples > 1
        num_files = 5 if not randomize else random.randint(2, 5)
        indentation = 4 if not randomize else random.choice([2, 4])
        structure_string = utils.data.show_project_structure(
            structure, indentation=indentation, randomize=randomize
        ).strip()
        message = utils.prompts.LOCALIZATION.format(
            problem_statement=problem_statement,
            structure=structure_string,
            n=num_files,
        ).strip()

        return message

    files, _classes, _functions = (
        utils.data.get_full_file_paths_and_classes_and_functions(structure)
    )
    all_found_files = list[list[str]]()
    all_traj = list[dict]()

    all_requests = [
        dict(
            model=args.model,
            messages=[dict(role="user", content=get_message())],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            n=1,
        )
        for _ in range(args.num_samples)
    ]

    idx_and_responses = await utils.api.collect_responses_async(
        client, semaphore, all_requests
    )
    assert len(idx_and_responses) == args.num_samples

    for idx, response in idx_and_responses:
        request = all_requests[idx]
        prompt = request["messages"][0]["content"]
        if response is not None:
            output: str = response.choices[0].message.content
        else:
            output = ""
        traj = dict(prompt=prompt, response=output)
        all_traj.append(traj)
        output = utils.api.parse_thinking_output(output)
        model_found_files = output.strip().split("\n")
        found_files = utils.data.correct_file_paths(model_found_files, files)
        all_found_files.append(found_files)

    return (all_found_files, all_traj)


async def main(
    bench_args: utils.args.BenchArgs,
    inference_args: utils.args.InferenceArgs,
    args: Args,
):
    output_folder = Path(args.output_file).parent
    output_folder.mkdir(parents=True, exist_ok=True)
    # write the arguments
    meta = dict(
        bench_args=str(bench_args),
        inference_args=str(inference_args),
        args=str(args),
    )
    with (output_folder / "args.json").open("w") as f:
        json.dump(meta, f, indent=4)

    swe_bench_data = bench_args.load()
    existing_instance_ids = utils.misc.load_existing_instance_ids(args.output_file)

    client = utils.api.OpenAIClient()
    semaphore = asyncio.Semaphore(inference_args.max_concurrent_requests)
    all_tasks = [
        localize_instance(client, semaphore, bug, inference_args, existing_instance_ids)
        for bug in swe_bench_data
    ]
    pbar = tqdm(total=len(all_tasks), desc="Process all instances")
    for completion in asyncio.as_completed(all_tasks):
        result = await completion
        if result is not None:
            with Path(args.output_file).open("a") as f:
                f.write(json.dumps(result) + "\n")
        pbar.update(1)


if __name__ == "__main__":
    params = utils.args.parse_args_into_dataclasses(
        utils.args.BenchArgs,
        utils.args.InferenceArgs,
        Args,
    )
    asyncio.run(main(*params))
