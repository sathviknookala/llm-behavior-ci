"""Validate one statistical method from a JSON spec.

Requires --spec and --output. Writes a public validation summary only.
Does not commit a null check or a results/ measurement. Bare invocation
exits 2.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from llm_behavior_ci.experiments.validation import main_validate


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    try:
        return main_validate(args_list)
    except SystemExit as error:
        code = error.code
        if code is None:
            return 2
        return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
