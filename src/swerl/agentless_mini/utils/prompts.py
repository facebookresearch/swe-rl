# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

from swerl.core.prompts import AGENTLESS_REPAIR as REPAIR

LOCALIZATION = """
Please look through a given GitHub issue and repository structure and provide a list of files that one would need to edit or look at to solve the issue.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}


###

Only provide the full path and return at most {n} files. The returned files should be separated by new lines ordered by most to least important and wrapped with ```. For example:

```
most/important/file1.xx
less/important/file2.yy
least/important/file3.zz
```
""".strip()
