"""Compare top-k and full plan KL on a supplied sample.

Requires --sample, --top-k, and --output. Bare invocation exits 2.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from llm_behavior_ci.experiments.validation import main_compare_kl


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    try:
        return main_compare_kl(args_list)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
