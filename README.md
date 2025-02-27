# SWE-RL

<p align="left">
    <a href="https://arxiv.org/abs/2502.18449"><img src="https://img.shields.io/badge/arXiv-2502.18449-b31b1b.svg?style=for-the-badge">
</p>

<p align="left">
    🧐&nbsp;<a href="#-about">About</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 🐣&nbsp;<a href="#-agentless-mini">Agentless Mini</a>
    | 📝&nbsp;<a href="#-citation">Citation</a>
    | 🙏&nbsp;<a href="#-acknowledgements">Acknowledgements</a>
</p>

## 🧐 About

Official codebase for our paper: **SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution** ([link](https://arxiv.org/abs/2502.18449)).

**SWE-RL** is the first approach to scale reinforcement learning based LLM reasoning for real-world software engineering, leveraging open-source software evolution data and rule-based rewards.

![Overview of SWE-RL](assets/swerl-overview.svg)

> [!NOTE]
> We have undertaken significant code refactoring to enhance quality and accessibility. However, this may introduce potential inconsistencies with our internal implementation. If you encounter a bug, please file an issue. We are also gradually updating the repo to include additional information.


## 🚀 Quick start

```bash
git clone https://github.com/facebookresearch/swe-rl && cd swe-rl
pip install -e ".[dev]"
pytest
```

The code currently provides our prompt templates and the implementation of the reward function based on sequence similarity.
You can find them in [src/swerl/core/prompts.py](src/swerl/core/prompts.py) and [src/swerl/core/reward.py](src/swerl/core/reward.py) respectively.

A toy example on how you can use the reward function in your own project:

``````python
import swerl

file = """
def sort_list(lst):
    return sorted(lst)
""".strip()

oracle_file = """
def sort_list(lst: list[int]) -> list[int]:
    return sorted(lst)
""".strip()

context = {"example.py": file}
oracle = {"example.py": oracle_file}

output = """
<think>
...thoughts by LLM
</think>
<solution>
```python
### example.py
<<<<<<< SEARCH
def sort_list(lst):
=======
def sort_list(lst: list[int]) -> list[int]:
>>>>>>> REPLACE
```
</solution>
""".strip()

reward, metadata = swerl.core.reward.calculate_search_replace_reward(context, oracle, output)
assert reward == 1.0
print(metadata)
``````

You can also check `swerl.core.reward.calculate_reward`, which is more general and can be paired with any editing format.

## 🐣 Agentless Mini

![Agentless Mini](assets/agentless-mini.svg)

Agentless Mini builds on top of [Agentless](https://github.com/OpenAutoCoder/Agentless) with the following key improvements and functionality changes:

1. Fast async inference with [openai-python](https://github.com/openai/openai-python).
2. Code refactoring for better scalability, parallelization, and accessibility.
3. Only performing file-level localization, and entire file content will be used for repair.
4. Support of using multiple reproduction tests for reranking.

### Environment setup

To get started, run the following command to install the dependencies:

```bash
git clone https://github.com/facebookresearch/swe-rl && cd swe-rl
pip install -e ".[agentless]"
```

Agentless Mini works with any OpenAI-compatible endpoint.
If you want to host your own Hugging Face models, popular choices are [vLLM](https://docs.vllm.ai/en/latest/) and [SGLang](https://docs.sglang.ai/). Taking vLLM as an example:

```bash
# Host Llama-3.3-70B-Instruct with vLLM
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 --port 8000
# The endpointn url will be http://localhost:8000/v1
```

Finally, you would need to set up some environment variables required by Agentless Mini:

```bash
# Assume you're doing the above vLLM setup
# Otherwise, just adjust them accordingly
export OPENAI_API_KEY="Empty"
export OPENAI_BASE_URL="http://localhost:8000/v1"

# Whether "thinking" is in model output (yes/no). If so, we need to extract the answer block during parsing
# and ignore the thinking. We assume the answer is enclosed with "<solution>" and "</solution>".
# Check src/swerl/agentless_mini/utils/envs.py to learn how to adjust them.
export THINKING=no
# A temporary directory used to process patches
export PLAYGROUND_DIR="tmp_agentless"
# Please download it from https://github.com/OpenAutoCoder/Agentless/releases/download/v1.5.0/swebench_repo_structure.txt
export PROJECT_FILE_LOC="/path/to/swebench/repo_structures"
# The tokenizer model. Can be either huggingface or tiktoken model name
export TOKENIZER_MODEL="meta-llama/Llama-3.3-70B-Instruct"
```

### Repair with oracle files

Now you can run Agentless Mini if the environment variables are properly configured.
Below is the simplest setup where oracle files are provided for repair. This can be a good proxy for end to end result:

```bash
# Make sure your are in the root directory of swe-rl
#
# Agentless Mini supports sharding. If you are using a compute cluster, then you can run
# different shards with different compute nodes to parallelize the evaluation.
# Below, we set num_shards to 125, so each shard will have (500 / 125) instances, where
# 500 is the number of problems in SWE-bench Verified.
python -m swerl.agentless_mini.repair \
    --loc_file resources/sweb_verified_gt_loc.jsonl \
    --output_folder demo_gt_repair \
    --shard 0 \
    --num_shards 125 \
    --num_samples 1 \
    --temperature 0.0 \
    --model "meta-llama/Llama-3.3-70B-Instruct"

# Get your all_preds.jsonl
python -m swerl.agentless_mini.rerank \
    --patch_folder ${REPAIR_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --output_file demo_gt_repair/all_preds.jsonl \
    --deduplicate
```

### Full pipeline

#### Localization + repair

You can also run the full pipeline. We show a greedy-decoding demo below:

```bash
NUM_SAMPLES=1
COMMON_ARGS=(
    --shard 0
    --num_shards 125
    --num_samples ${NUM_SAMPLES}
    --temperature 0.0
    --model "meta-llama/Llama-3.3-70B-Instruct"
    # Check --max_concurrent_requests on how to control the concurrency
)

ROOT=demo_agentless
LOC_FILE=${ROOT}/loc.jsonl
REPAIR_DIR=${ROOT}/repair
PRED_FILE=${ROOT}/all_preds.jsonl

# Localization
python -m swerl.agentless_mini.localize \
    --output_file ${LOC_FILE} \
    ${COMMON_ARGS[@]}

# Optionally, check localization performance
python -m swerl.agentless_mini.tools.check_loc_perf --locfile ${LOC_FILE}

# Repair
python -m swerl.agentless_mini.repair \
    --loc_file ${LOC_FILE} \
    --output_folder ${REPAIR_DIR} \
    ${COMMON_ARGS[@]}

# Rerank
python -m swerl.agentless_mini.rerank \
    --patch_folder ${REPAIR_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --output_file ${PRED_FILE} \
    --deduplicate

# Now the ${PRED_FILE} will be ready. If you get all empty outputs, it means
# the model isn't generating correctly formatted edits. Then you should consider
# changing your base model or sampling more locations & repairs.
```

#### Reproduction test generation

> [!NOTE]
> Reproduction test generation, regression test selection, and test execution are WIP due to refactoring and infra difference.
> They will be updated shortly.


## 📝 Citation

```bibtex
@article{wei2025swerl,
  title={SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution}, 
  author={Yuxiang Wei and Olivier Duchenne and Jade Copet and Quentin Carbonneaux and Lingming Zhang and Daniel Fried and Gabriel Synnaeve and Rishabh Singh and Sida I. Wang},
  year={2025},
  journal={arXiv preprint arXiv:2502.18449}
}
```

## 🙏 Acknowledgements

[Agentless](https://github.com/OpenAutoCoder/Agentless),
[SWE-Gym](https://github.com/SWE-Gym/SWE-Gym),
[SWE-Fixer](https://github.com/InternLM/SWE-Fixer),
[SWE-bench](https://github.com/SWE-bench/SWE-bench),
[Moatless EvalTools](https://eval.moatless.ai/),
[Nebius SWE-agent](https://nebius.com/blog/posts/training-and-search-for-software-engineering-agents).

## License

The majority of SWE-RL is licensed under CC BY-NC 4.0, however portions of the project are available under separate license terms: [Agentless Mini](src/swerl/agentless_mini) is licensed under the MIT license.
