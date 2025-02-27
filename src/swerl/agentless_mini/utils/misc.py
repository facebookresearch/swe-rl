# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import json
import os
from pathlib import Path

import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizer

from .envs import TOKENIZER_MODEL, TOKENIZER_TYPE


def load_jsonl(filepath: str | Path) -> list[dict]:
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


def write_jsonl(data: list[dict], filepath: str | Path):
    """
    Write data to a JSONL file at the given filepath.

    Arguments:
    data -- a list of dictionaries to write to the JSONL file
    filepath -- the path to the JSONL file to write
    """
    with open(filepath, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def load_json(filepath: str) -> dict:
    return json.load(open(filepath, "r"))


def load_existing_instance_ids(output_file: str) -> set[str]:
    instance_ids = set[str]()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    instance_ids.add(data["instance_id"])
                except json.JSONDecodeError:
                    continue
    return instance_ids


_TOKENIZER = None


def get_tokenizer() -> PreTrainedTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    return _TOKENIZER


def count_tokens(messages_or_prompt: list[dict] | str) -> int:
    """Count tokens for the specified tokenizer."""
    if TOKENIZER_TYPE == "hf":
        return count_hf_tokens(messages_or_prompt)
    return count_tiktoken_tokens(messages_or_prompt)


def count_hf_tokens(messages_or_prompt: list[dict] | str) -> int:
    """Count tokens for HF tokenizer."""
    tokenizer = get_tokenizer()
    if isinstance(messages_or_prompt, str):
        return len(tokenizer.encode(messages_or_prompt))
    return len(
        tokenizer.apply_chat_template(messages_or_prompt, add_generation_prompt=True)
    )


def count_tiktoken_tokens(messages: list[dict] | str) -> int:
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    if isinstance(messages, str):
        return len(encoding.encode(messages))
    num_tokens = sum(len(encoding.encode(message["content"])) for message in messages)
    return num_tokens
