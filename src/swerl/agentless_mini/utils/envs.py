# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""Defines the environment variables that are used by the agentless module."""

import os
from typing import cast

# Whether or not to enable the thinking mode (yes or no)
THINKING = os.environ.get("THINKING", "no") == "yes"
# If enabled, what's the enclosing tag for fetching the answer from output
ANSWER_START_TAG = os.environ.get("ANSWER_START_TAG", "<solution>")
ANSWER_END_TAG = os.environ.get("ANSWER_END_TAG", "</solution>")
# Where to put temporary generated files during processing & reranking
PLAYGROUND_DIR = os.getenv("PLAYGROUND_DIR", "playground")
# Preprocessed structure information for each SWE-Bench problem
# Please download it from the original Agentless repository
# https://github.com/OpenAutoCoder/Agentless/tree/main
PROJECT_FILE_LOC = cast(str, os.environ.get("PROJECT_FILE_LOC", None))

assert (
    PROJECT_FILE_LOC is not None
), f"""PROJECT_FILE_LOC must be set.
Refer to the original Agentless repo to download the files or construct them from scratch"""

# The path to the HF tokenizer model for counting context tokens
# Or tiktoken model name
TOKENIZER_MODEL = cast(str, os.getenv("TOKENIZER_MODEL", None))
assert TOKENIZER_MODEL is not None
# The tokenizer type to use for counting tokens
# hf or tiktoken
TOKENIZER_TYPE = os.getenv("TOKENIZER_TYPE", "hf")
assert TOKENIZER_TYPE in ["hf", "tiktoken"], f"Invalid TOKENIZER_TYPE: {TOKENIZER_TYPE}"
