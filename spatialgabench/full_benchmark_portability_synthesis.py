"""FB2 seven-study construction portability synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    loader = json.loads if path.suffix == ".json" else yaml.safe_load
    value = loader(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _validate_authorization(
    config: dict[str, Any], config_path: Path, authorization_path: Path, root: Path
) -> dict[str, Any]:
    authorization = _load(authorization_path)
    required_false = (
        "construction_rerun_authorized",
        "policy_refit_authorized",
        "resource_ranking_authorized",
        "universal_ranking_authorized",
        "final_ranking_authorized",
    )
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "FB2_seven_study_portability_synthesis"
        or authorization.get("full_benchmark_expansion_authorized") is not True
        or any(authorization.get(key) is not False for key in required_false)
    ):
        raise PermissionError("FB2 authorization boundary is invalid")
    bindings = {
        "config": config_path,
        "implementation": Path(__file__).resolve(),
        "completion_protocol": root / config["completion_protocol"],
        "development_input": root / config["development_input"],
        "development_gate": root / config["development_gate"],
        "external_input": root / config["external_input"],
        "external_gate": root / config["external_gate"],
    }
    for key, path in bindings.items():
        record = authorization.get(key, {})
        if (
            not path.is_file()
            or record.get("path") != str(path.relative_to(root))
            or record.get("sha256") != sha256(path)
        ):
            raise ValueError(f"FB2 authorization binding mismatch: {key}")
    if float(config["confidence_level"]) != 0.95:
        raise ValueError("FB2 confidence level changed")
    return authorization


def _common_pairs(config: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = {
        (representation, method)
        for representation in config["matrix_representations"]
        for method in config["annotation_methods"]
    }
    pairs.update(
        (config["fragment_representation"], method) for method in config["software_methods"]
    )
    return pairs


def _development_manifest(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pairs = _common_pairs(config)
    selected = frame[
        frame.apply(lambda row: (row.representation_id, row.method_id) in pairs, axis=1)
    ].copy()
    selected["study_role"] = "development"
    selected["source_record"] = config["development_input"]
    selected["construction_status"] = selected["run_status"].astype(str)
    selected["evaluation_status"] = "not_run_portability_is_construction_endpoint"
    selected["success"] = (
        selected["eligible"].eq(True)
        & selected["run_status"].eq("pass")
        & selected["valid_output_indicator"].eq(True)
        & selected["output_contract_pass"].eq(True)
    )
    selected["failure_reason"] = selected["not_applicable_reason"].fillna("")
    columns = [
        "study_id", "study_role", "dataset_id", "sample_id", "species", "genome_build",
        "representation_id", "input_track", "method_id", "method_category", "subset_only",
        "eligible", "construction_status", "evaluation_status", "success", "failure_reason",
        "resource_comparison_allowed", "resource_scope", "source_record",
    ]
    return selected[columns]


def _external_manifest(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selected = frame[frame.task_id.eq("portability_runtime_failure")].copy()
    metadata = config["external_studies"]
    selected["study_role"] = "external"
    selected["dataset_id"] = ""
    selected["sample_id"] = ""
    selected["species"] = selected.study_id.map(lambda x: metadata[x]["species"])
    selected["genome_build"] = selected.study_id.map(lambda x: metadata[x]["genome_build"])
    selected["input_track"] = selected.representation_id.map(
        lambda x: "F" if x == config["fragment_representation"] else ("P" if x == "macs3_summit_501bp" else "B")
    )
    selected["method_category"] = selected.method_id.map(
        lambda x: "software" if x in config["software_methods"] else "annotation"
    )
    selected["subset_only"] = False
    selected["eligible"] = selected["admissible"].eq(True)
    selected["success"] = (
        selected["eligible"]
        & selected["construction_status"].eq("pass")
        & selected["evaluation_status"].eq("pass")
    )
    selected["failure_reason"] = ""
    selected["resource_comparison_allowed"] = False
    selected["resource_scope"] = "not_collected_in_common_comparative_protocol"
    selected["source_record"] = config["external_input"]
    columns = [
        "study_id", "study_role", "dataset_id", "sample_id", "species", "genome_build",
        "representation_id", "input_track", "method_id", "method_category", "subset_only",
        "eligible", "construction_status", "evaluation_status", "success", "failure_reason",
        "resource_comparison_allowed", "resource_scope", "source_record",
    ]
    return selected[columns]


def _summary_row(scope: str, value: str, frame: pd.DataFrame) -> dict[str, Any]:
    total = len(frame)
    successes = int(frame.success.sum())
    low, high = wilson_interval(successes, total)
    return {
        "grouping_scope": scope,
        "grouping_value": value,
        "eligible_contracts": total,
        "successful_contracts": successes,
        "failed_contracts": total - successes,
        "success_rate": successes / total if total else math.nan,
        "wilson_95_ci_low": low,
        "wilson_95_ci_high": high,
        "all_pass_ceiling": bool(total and successes == total),
    }


def _success_summary(common: pd.DataFrame) -> pd.DataFrame:
    rows = [_summary_row("overall", "all", common)]
    for scope, column in (
        ("study_role", "study_role"),
        ("species_genome_build", "species_build"),
        ("study", "study_id"),
        ("method", "method_id"),
        ("representation", "representation_id"),
        ("input_track", "input_track"),
    ):
        source = common.copy()
        if column == "species_build":
            source[column] = source.species + "/" + source.genome_build
        for value, group in source.groupby(column, sort=True):
            rows.append(_summary_row(scope, str(value), group))
    return pd.DataFrame(rows)


def _write_table(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    frame.to_csv(path, sep="\t", index=False)
    return {"path": str(path), "rows": len(frame), "sha256": sha256(path)}


def run(config_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    config = _load(config_path)
    _validate_authorization(config, config_path, authorization_path, root)

    development_raw = pd.read_csv(root / config["development_input"], sep="\t")
    external_raw = pd.read_csv(root / config["external_input"], sep="\t")
    development = _development_manifest(development_raw, config)
    external = _external_manifest(external_raw, config)
    common = pd.concat([development, external], ignore_index=True).sort_values(
        ["study_id", "representation_id", "method_id"], kind="mergesort"
    ).reset_index(drop=True)
    expected = int(config["expected_common_rows"])
    per_study = common.groupby("study_id").size()
    if (
        len(common) != expected
        or common.study_id.nunique() != 7
        or not per_study.eq(int(config["expected_common_rows_per_study"])).all()
        or common.duplicated(["study_id", "representation_id", "method_id"]).any()
    ):
        raise ValueError("FB2 common panel is incomplete or duplicated")

    summary = _success_summary(common)
    failure = (
        common.assign(
            outcome=common.success.map({True: "success", False: "failure"}),
            failure_reason=common.failure_reason.replace("", "none"),
        )
        .groupby(["study_role", "outcome", "construction_status", "failure_reason"], dropna=False)
        .size().rename("contracts").reset_index()
    )
    coverage = common.pivot(
        index=["representation_id", "method_id"], columns="study_id", values="success"
    ).reset_index()
    coverage["studies_covered"] = coverage.drop(columns=["representation_id", "method_id"]).sum(axis=1)
    coverage["all_seven_studies_pass"] = coverage.studies_covered.eq(7)

    cicero = development_raw[
        development_raw.method_id.eq(config["subset_method"])
        & development_raw.representation_id.eq(config["subset_representation"])
    ].copy()
    cicero = cicero[[
        "study_id", "dataset_id", "sample_id", "species", "genome_build", "representation_id",
        "method_id", "requested_cells", "eligible", "run_status", "not_applicable_reason",
        "runtime_seconds_observed", "peak_aggregate_pss_mb_observed", "resource_comparison_allowed",
        "resource_scope",
    ]]
    resources = development_raw[
        development_raw.apply(lambda row: (row.representation_id, row.method_id) in _common_pairs(config), axis=1)
    ][[
        "study_id", "representation_id", "method_id", "runtime_seconds_observed",
        "peak_aggregate_pss_mb_observed", "peak_aggregate_rss_mb_observed",
        "resource_comparison_allowed", "resource_scope",
    ]].copy()
    resources["comparative_interpretation"] = "forbidden_noncommon_nonisolated_measurement_protocol"

    ceiling = summary[summary.grouping_scope.isin(["overall", "study_role", "species_genome_build"])].copy()
    ceiling["diagnosis"] = ceiling.all_pass_ceiling.map(
        {True: "nondiscriminating_all_pass_ceiling", False: "failures_observed"}
    )
    ceiling["superiority_inference_allowed"] = False

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, frame in (
        ("common_contract_manifest.tsv", common),
        ("success_rate_wilson_summary.tsv", summary),
        ("failure_accounting.tsv", failure),
        ("seven_study_coverage_matrix.tsv", coverage),
        ("subset_cicero_accounting.tsv", cicero),
        ("resource_observations_noncomparative.tsv", resources),
        ("ceiling_diagnosis.tsv", ceiling),
    ):
        path = output / name
        artifacts[name] = _write_table(frame, path)
        artifacts[name]["path"] = str(path.relative_to(root))

    evidence = {
        "schema_version": 1,
        "gate": "FULL_BENCHMARK",
        "milestone": "FB2_seven_study_portability_synthesis",
        "status": "pass",
        "study_families": int(common.study_id.nunique()),
        "development_study_families": int(development.study_id.nunique()),
        "external_study_families": int(external.study_id.nunique()),
        "human_hg38_study_families": int(common.loc[common.genome_build.eq("hg38"), "study_id"].nunique()),
        "mouse_mm10_study_families": int(common.loc[common.genome_build.eq("mm10"), "study_id"].nunique()),
        "common_contract_rows": len(common),
        "successful_contract_rows": int(common.success.sum()),
        "failed_contract_rows": int((~common.success).sum()),
        "common_methods": int(common.method_id.nunique()),
        "common_representations": int(common.representation_id.nunique()),
        "all_pass_ceiling": bool(common.success.all()),
        "portability_status": "supported_nondiscriminating_ceiling" if common.success.all() else "failures_observed",
        "method_superiority_inference_allowed": False,
        "resource_measurements_comparative": False,
        "subset_cicero_in_common_denominator": False,
        "construction_rerun": False,
        "policy_refit_run": False,
        "resource_ranking_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "artifacts": artifacts,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256(authorization_path)},
        "implementation": {"path": "spatialgabench/full_benchmark_portability_synthesis.py", "sha256": sha256(Path(__file__))},
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_benchmark_fb2_portability_synthesis.yaml")
    parser.add_argument("--authorization", default="governance/authorizations/FULL_BENCHMARK_FB2_PORTABILITY_SYNTHESIS_AUTHORIZATION_2026-07-18.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(run(args.config, args.authorization, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
