"""QC-driven recommendation framework for SpatialGABench."""

from __future__ import annotations

from spatialgabench.recommendation.engine import recommend_methods
from spatialgabench.recommendation.frozen import load_frozen_policy, recommend_from_frozen_policy
from spatialgabench.recommendation.report import write_recommendation_report
from spatialgabench.recommendation.schema import (
    load_recommendation_input,
    validate_recommendation_input,
)


def default_recommendation_policy() -> dict[str, str]:
    """Return the public high-level recommendation policy."""

    return {
        "ranking_policy": "no_universal_ranking",
        "comparison_policy": "representation_aware",
        "rna_policy": "evaluation_proxy_not_ground_truth",
        "software_policy": "input_contract_aware",
    }


__all__ = [
    "default_recommendation_policy",
    "load_recommendation_input",
    "load_frozen_policy",
    "recommend_methods",
    "recommend_from_frozen_policy",
    "validate_recommendation_input",
    "write_recommendation_report",
]
