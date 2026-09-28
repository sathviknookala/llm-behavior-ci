from dataclasses import asdict, dataclass
import json
import math

from llm_behavior_ci.stats.bootstrap import (
    PairedBootstrapResult,
    paired_bootstrap,
)
from llm_behavior_ci.stats.kl import NextTokenKLResult, next_token_kl
from llm_behavior_ci.stats.mmd import MMDResult, mmd_permutation_test


DEMO_PRODUCTION_SCORES = (1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
DEMO_CANDIDATE_SCORES = (1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
DEMO_PRODUCTION_NEXT_TOKEN_PROBABILITIES = (
    (0.70, 0.20, 0.10),
    (0.10, 0.80, 0.10),
    (0.25, 0.25, 0.50),
)
DEMO_CANDIDATE_NEXT_TOKEN_PROBABILITIES = (
    (0.68, 0.22, 0.10),
    (0.11, 0.78, 0.11),
    (0.24, 0.26, 0.50),
)
DEMO_PRODUCTION_EMBEDDINGS = (
    (0.0, 0.0),
    (0.2, 0.1),
    (0.4, 0.2),
    (0.6, 0.3),
    (0.8, 0.4),
    (1.0, 0.5),
)
DEMO_CANDIDATE_EMBEDDINGS = DEMO_PRODUCTION_EMBEDDINGS
DEMO_CONFIDENCE_LEVEL = 0.95
DEMO_BOOTSTRAP_RESAMPLES = 2_000
DEMO_MINIMUM_SCORE_DELTA = -0.02
DEMO_MAXIMUM_MEAN_KL_NATS = 0.01
DEMO_MMD_BANDWIDTH = 1.0
DEMO_MMD_PERMUTATIONS = 499
DEMO_MMD_ALPHA = 0.05
DEMO_SEED = 20260926


@dataclass(frozen=True)
class OfflineGateResult:
    bootstrap: PairedBootstrapResult
    next_token_kl: NextTokenKLResult
    mmd: MMDResult
    bootstrap_passed: bool
    next_token_kl_passed: bool
    mmd_passed: bool

    @property
    def passed(self) -> bool:
        return (
            self.bootstrap_passed
            and self.next_token_kl_passed
            and self.mmd_passed
        )


def _log_probabilities(
    positions: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(math.log(probability) for probability in position)
        for position in positions
    )


def run_offline_gate() -> OfflineGateResult:
    bootstrap = paired_bootstrap(
        candidate=DEMO_CANDIDATE_SCORES,
        production=DEMO_PRODUCTION_SCORES,
        confidence_level=DEMO_CONFIDENCE_LEVEL,
        resamples=DEMO_BOOTSTRAP_RESAMPLES,
        seed=DEMO_SEED,
    )
    kl = next_token_kl(
        production_log_probabilities=_log_probabilities(
            DEMO_PRODUCTION_NEXT_TOKEN_PROBABILITIES
        ),
        candidate_log_probabilities=_log_probabilities(
            DEMO_CANDIDATE_NEXT_TOKEN_PROBABILITIES
        ),
    )
    mmd = mmd_permutation_test(
        production=DEMO_PRODUCTION_EMBEDDINGS,
        candidate=DEMO_CANDIDATE_EMBEDDINGS,
        bandwidth=DEMO_MMD_BANDWIDTH,
        permutations=DEMO_MMD_PERMUTATIONS,
        seed=DEMO_SEED,
    )

    return OfflineGateResult(
        bootstrap=bootstrap,
        next_token_kl=kl,
        mmd=mmd,
        bootstrap_passed=(
            bootstrap.confidence_low >= DEMO_MINIMUM_SCORE_DELTA
        ),
        next_token_kl_passed=(
            kl.mean_kl_nats <= DEMO_MAXIMUM_MEAN_KL_NATS
        ),
        mmd_passed=mmd.p_value > DEMO_MMD_ALPHA,
    )


def result_payload(result: OfflineGateResult) -> dict[str, object]:
    return {
        "passed": result.passed,
        "checks": {
            "paired_bootstrap": {
                **asdict(result.bootstrap),
                "passed": result.bootstrap_passed,
                "minimum_score_delta": DEMO_MINIMUM_SCORE_DELTA,
            },
            "next_token_kl": {
                **asdict(result.next_token_kl),
                "passed": result.next_token_kl_passed,
                "maximum_mean_kl_nats": DEMO_MAXIMUM_MEAN_KL_NATS,
            },
            "mmd": {
                **asdict(result.mmd),
                "passed": result.mmd_passed,
                "alpha": DEMO_MMD_ALPHA,
            },
        },
    }


def main() -> int:
    result = run_offline_gate()
    print(json.dumps(result_payload(result), indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
