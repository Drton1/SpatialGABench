"""Exact-contract access to the frozen G4 recommendation policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_POLICY = Path("results/v1/g4/m4_recommendation_policy/frozen_task_specific_recommendation_policy.json")
FORBIDDEN_CONTEXT_INPUTS = {
    "rna_agreement_outcome",
    "labels",
    "clusters",
    "pathology_outcome",
    "simulation_truth",
    "holdout_method_outcome",
}


def load_frozen_policy(path: str | Path = DEFAULT_POLICY) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "frozen":
        raise ValueError("Recommendation policy is not a frozen JSON mapping")
    if payload.get("holdout_accessed") is not False or payload.get("universal_ranking") is not False:
        raise ValueError("Frozen recommendation policy boundary is invalid")
    return payload


def recommend_from_frozen_policy(
    *,
    primary_task: str,
    input_track: str,
    representation_id: str,
    species: str,
    genome_build: str,
    fragments_available: bool,
    policy_path: str | Path = DEFAULT_POLICY,
    **context: Any,
) -> dict[str, Any]:
    """Return one frozen leaf; outcomes and fallback matching are forbidden."""
    forbidden = sorted(FORBIDDEN_CONTEXT_INPUTS & set(context))
    if forbidden:
        raise ValueError(f"Evaluation outcomes cannot enter policy selection: {forbidden}")
    if input_track not in {"F", "P", "B"}:
        raise ValueError("input_track must be one of F, P, or B")
    if input_track == "F" and fragments_available is not True:
        raise ValueError("F-track recommendation requires fragments_available=true")
    if species not in {"mouse", "human"} or genome_build not in {"mm10", "hg38"}:
        raise ValueError("The frozen compatibility contract currently supports mouse/mm10 and human/hg38")

    policy = load_frozen_policy(policy_path)
    exact = [
        leaf
        for leaf in policy["leaves"]
        if leaf["status"] == "active_frozen"
        and leaf["task_id"] == primary_task
        and leaf["input_track"] == input_track
        and leaf["representation_scope"] == representation_id
    ]
    if len(exact) != 1:
        task_status = [leaf for leaf in policy["leaves"] if leaf["task_id"] == primary_task]
        return {
            "status": "no_frozen_leaf",
            "primary_task": primary_task,
            "input_track": input_track,
            "representation_id": representation_id,
            "recommended_methods": [],
            "reason": (
                task_status[0]["required_caveat"]
                if len(task_status) == 1 and task_status[0]["status"] != "active_frozen"
                else "No exact development-supported task/track/representation leaf is frozen; fallback matching is forbidden."
            ),
            "universal_fallback_used": False,
        }
    leaf = exact[0]
    support_species = [value for value in str(leaf["support_species"]).split(";") if value]
    support_genomes = [value for value in str(leaf["support_genome_builds"]).split(";") if value]
    if species not in support_species or genome_build not in support_genomes:
        return {
            "status": "no_frozen_leaf",
            "primary_task": primary_task,
            "input_track": input_track,
            "representation_id": representation_id,
            "species": species,
            "genome_build": genome_build,
            "recommended_methods": [],
            "reason": "The exact leaf has no development evidence for the requested species/genome; compatibility alone cannot establish task utility.",
            "support_species": support_species,
            "support_genome_builds": support_genomes,
            "universal_fallback_used": False,
        }
    methods = [method for method in str(leaf["recommended_methods"]).split(";") if method]
    return {
        "status": "frozen_leaf_match",
        "policy_id": policy["policy_id"],
        "leaf_id": leaf["policy_id"],
        "primary_task": primary_task,
        "input_track": input_track,
        "representation_id": representation_id,
        "species": species,
        "genome_build": genome_build,
        "recommended_methods": methods,
        "required_controls": [value for value in str(leaf["required_controls"]).split(";") if value],
        "support_studies": [value for value in str(leaf["support_studies"]).split(";") if value],
        "support_species": support_species,
        "support_genome_builds": support_genomes,
        "selection_rule": leaf["selection_rule"],
        "required_caveat": leaf["required_caveat"],
        "forbidden_interpretation": leaf["forbidden_interpretation"],
        "universal_fallback_used": False,
        "holdout_outcome_used": False,
    }
