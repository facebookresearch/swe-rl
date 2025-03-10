# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

from swerl.core.reward import calculate_reward_unidiff, calculate_search_replace_reward

FILE_A = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
""".strip()

FILE_A_ORACLE = """
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b
""".strip()

FILE_B = """
def sort_list(lst):
    return sorted(lst)
""".strip()

FILE_B_ORACLE = """
def sort_list(lst: list[int]) -> list[int]:
    return sorted(lst)
""".strip()

CODE_CONTEXT = {
    "a.py": FILE_A,
    "b.py": FILE_B,
}

ORACLE_NEW_CONTENT_CHANGE_0 = {
    "a.py": FILE_A,
    "b.py": FILE_B,
}

ORACLE_NEW_CONTENT_CHANGE_1 = {
    "a.py": FILE_A,
    "b.py": FILE_B_ORACLE,
}

ORACLE_NEW_CONTENT_CHANGE_2 = {
    "a.py": FILE_A_ORACLE,
    "b.py": FILE_B_ORACLE,
}


def test_multifile_edit():
    output = """
<think>
dummy
</think>
<solution>
```
### a.py
<<<<<<< SEARCH
def add(a, b):
=======
def add(a: int, b: int) -> int:
>>>>>>> REPLACE
```

```python
### a.py
<<<<<<< SEARCH
def subtract(a, b):
=======
def subtract(a: float, b: float) -> float:
>>>>>>> REPLACE
```

```python
### b.py
<<<<<<< SEARCH
def sort_list(lst):
=======
def sort_list(lst: list[int]) -> list[int]:
>>>>>>> REPLACE
```
</solution>
"""
    reward, meta = calculate_search_replace_reward(
        code_context=CODE_CONTEXT,
        oracle_new_content=ORACLE_NEW_CONTENT_CHANGE_2,
        output=output,
    )
    print(meta["similarities"])
    assert reward == 1.0, meta["similarities"]
    assert meta["thought"] == "dummy"


def test_zero_reward():
    output = """
<think>
dummy
</think>
<solution>
```
### c.py
<<<<<<< SEARCH

=======
# dummy
>>>>>>> REPLACE
```
</solution>
"""
    reward, meta = calculate_search_replace_reward(
        code_context=CODE_CONTEXT,
        oracle_new_content=ORACLE_NEW_CONTENT_CHANGE_2,
        output=output,
    )
    assert reward == 0.0, meta


def test_new_file():
    output = """
<think>
dummy
</think>
<solution>
```
### c.py
<<<<<<< SEARCH

=======
# dummy
>>>>>>> REPLACE
```

```
### c.py
<<<<<<< SEARCH
# dummy
=======
print("hello")
>>>>>>> REPLACE
```

```
### b.py
<<<<<<< SEARCH
def sort_list(lst):
=======
def sort_list(lst: list[int]) -> list[int]:
>>>>>>> REPLACE
```
</solution>
"""
    reward, meta = calculate_search_replace_reward(
        code_context=CODE_CONTEXT,
        oracle_new_content=ORACLE_NEW_CONTENT_CHANGE_2,
        output=output,
    )
    # b is 1, the others are 0
    assert reward == 1 / 3, meta


def test_failed_parse():
    output1 = """
<think>
dummy
</think>
<solution>
"""
    output2 = """
<think>
dummy
</think>
<solution>
```
### a.py
<<<<<<< SEARCH
return a + b
=======
return a - b
```
</solution>"""
    for output in [output1, output2]:
        reward, meta = calculate_search_replace_reward(
            code_context=CODE_CONTEXT,
            oracle_new_content=ORACLE_NEW_CONTENT_CHANGE_2,
            output=output,
        )
        assert reward == -1.0, meta


DIFF_1 = """diff --git a/foo.txt b/foo.txt
index 83db48f..f73504f 100644
--- a/foo.txt
+++ b/foo.txt
@@ -1,4 +1,4 @@
-Hello world
+Hello World
 This is a file
-With some lines
+With some changed lines
 End
"""


DIFF_2 = """diff --git a/foo.txt b/foo.txt
index 83db48f..f73504f 100644
--- a/foo.txt
+++ b/foo.txt
@@ -1,4 +1,4 @@
-Hello world
+Hello World
 This is a file
-With some lines
+With some changed
 End
"""


DIFF_3 = """diff --git a/bar.txt b/bar.txt
index 83db48f..f73504f 100644
--- a/bar.txt
+++ b/bar.txt
@@ -1,4 +1,4 @@
-Hello world
+Hello World
 This is a file
-With some lines
+With some changed
 End
"""


def test_reward_unidiff():
    reward, meta = calculate_reward_unidiff(
        oracle_patches=[DIFF_1],
        pred_patches=[DIFF_2],
    )
    assert reward > 0.9, meta


def test_reward_unidiff_same():
    reward, meta = calculate_reward_unidiff(
        oracle_patches=[DIFF_1],
        pred_patches=[DIFF_1],
    )
    assert reward == 1.0, meta


def test_reward_unidiff_different():
    reward, meta = calculate_reward_unidiff(
        oracle_patches=[DIFF_1],
        pred_patches=[DIFF_3],
    )
    assert reward == 0.0, meta


def test_reward_unidiff_half():
    reward, meta = calculate_reward_unidiff(
        oracle_patches=[DIFF_1],
        pred_patches=[DIFF_1, DIFF_3],
    )
    assert reward == 0.5, meta
