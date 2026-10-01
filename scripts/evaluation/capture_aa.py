"""Capture paired A/A episodes.

Requires --configuration, --task-set, --stream-settings, --repetitions,
--concurrency, --modes, --output, and --vllm-base-url. Rejects test_normal.
Does not apply a protocol threshold. Bare invocation exits 2. Output under
results/ is refused. Refuses a git_commit that is not HEAD and a dirty
tracked source or config tree.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from llm_behavior_ci.runtime.aa_capture import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    try:
        return _main(args_list)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
