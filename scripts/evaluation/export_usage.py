"""Export public hosted-usage aggregates from a local episode store.

Requires --store (a local SQLite episode log) and --output. Every
finished episode in the store must share one configuration hash and one
split. --pricing is an optional versioned pricing JSON
(``llm_behavior_ci.usage``); without it, or when a priced count is
unknown, cost stays null. The output holds counts, latency, and cost only:
no task ids, prompts, outputs, reasoning text, or keys. Bare invocation
exits 2.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_behavior_ci.export import AggregateResults, ExportError, export_public_results
from llm_behavior_ci.storage import EpisodeStore, StorageError
from llm_behavior_ci.usage import UsageError, aggregate_usage, load_pricing


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return 2
    parser = argparse.ArgumentParser(description="Export hosted usage aggregates.")
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pricing", default=None)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as error:
        return 2 if error.code is None else int(error.code)
    store_path = Path(args.store)
    if not store_path.is_file():
        print("store does not exist", file=sys.stderr)
        return 1
    try:
        store = EpisodeStore(store_path)
        try:
            episodes = store.load_finished_episodes()
        finally:
            store.close()
        if not episodes:
            raise UsageError("store has no finished episodes")
        hashes = {episode.run.configuration_hash for episode in episodes}
        splits = {episode.task.split for episode in episodes}
        if len(hashes) != 1 or len(splits) != 1:
            raise UsageError("store mixes configurations or splits; export one at a time")
        pricing = None if args.pricing is None else load_pricing(Path(args.pricing))
        usage = aggregate_usage(
            episodes,
            split=splits.pop(),
            configuration_hash=hashes.pop(),
            pricing=pricing,
        )
        export_public_results(
            AggregateResults(aggregates=(), evidence=(), decisions=(), usage=usage),
            output_path=Path(args.output),
        )
    except (UsageError, StorageError, ExportError) as error:
        print(str(error) or "usage export failed", file=sys.stderr)
        return 1
    print(f"usage aggregates: {len(usage)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
