"""Rule definitions for the SpatialGABench recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


CORE_BASELINES = [
    "promoter_only_2kb_pc",
    "gene_body_pc",
    "promoter_plus_gene_body_2kb_pc",
]
DISTAL_METHODS = [
    "distance_weighted_distal_250kb_decay50kb_pc",
    "promoter_plus_distal_2kb_250kb_decay50kb_pc",
]
NEGATIVE_CONTROLS = [
    "random_peak_to_gene_negative_control_pc",
    "accessibility_matched_random_peak_to_gene_control_pc",
]
REPRESENTATION_BASELINES = ["raw_peak_lsi_baseline"]
SOFTWARE_FRAGMENT_METHODS = [
    "signac_geneactivity_mm10_or_hg38",
    "snapatac2_geneactivity_mm10_or_hg38",
    "archr_genescorematrix_mm10_or_hg38_policy_B",
]
CICERO_SUBSET = "cicero_geneactivity_mm10_or_hg38_subset_only"


@dataclass(frozen=True)
class RecommendationRule:
    """Single evidence-backed recommendation rule."""

    rule_id: str
    scenario: str
    trigger: Callable[[dict[str, Any]], bool]
    recommended_methods: tuple[str, ...] = ()
    methods_to_use_with_caution: tuple[str, ...] = ()
    not_applicable_methods: tuple[str, ...] = ()
    required_baselines: tuple[str, ...] = ()
    required_negative_controls: tuple[str, ...] = ()
    evaluation_tasks_to_run: tuple[str, ...] = ()
    comparison_scope: tuple[str, ...] = ()
    required_caveats: tuple[str, ...] = ()
    forbidden_interpretations: tuple[str, ...] = ()
    evidence_source: tuple[str, ...] = ()
    claim_strength: str = "context"


def _no_fragments(config: dict[str, Any]) -> bool:
    return not bool(config["fragments_available"]) and config["input_representation"] in {
        "author_peak_matrix",
        "fixed_bin_matrix",
    }


def _has_fragments(config: dict[str, Any]) -> bool:
    return bool(config["fragments_available"]) or config["input_representation"] == "fragments"


def _high_sparsity_low_depth(config: dict[str, Any]) -> bool:
    return config["data_sparsity"] == "high" or config["sequencing_depth"] == "low"


def _regulatory_context(config: dict[str, Any]) -> bool:
    return config["main_task"] == "regulatory_interpretation" or config["expected_distal_regulation"] == "high"


def _rna_task(config: dict[str, Any]) -> bool:
    return config["main_task"] == "rna_agreement"


def _cross_dataset(config: dict[str, Any]) -> bool:
    return config["main_task"] == "cross_dataset_benchmark"


def _simulation_truth(config: dict[str, Any]) -> bool:
    return config["main_task"] == "simulation_truth_recovery"


def _software_portability(config: dict[str, Any]) -> bool:
    return config["main_task"] == "software_portability" or bool(config["need_software_standard_workflow"])


def _missing_annotation(config: dict[str, Any]) -> bool:
    return not bool(config["has_gene_annotation"])


def _spatial_task_missing_spatial(config: dict[str, Any]) -> bool:
    return config["main_task"] in {"clustering_domain", "marker_recovery"} and not bool(
        config["has_spatial_coordinates"]
    )


RULES = [
    RecommendationRule(
        rule_id="R01_peak_matrix_no_fragments",
        scenario="Only peak/bin matrix is available and fragments are absent.",
        trigger=_no_fragments,
        recommended_methods=tuple(CORE_BASELINES),
        methods_to_use_with_caution=tuple(DISTAL_METHODS),
        not_applicable_methods=(
            "Signac fragment route",
            "ArchR fragment route",
            "SnapATAC2 fragment route",
        ),
        required_baselines=tuple(REPRESENTATION_BASELINES),
        required_negative_controls=tuple(NEGATIVE_CONTROLS),
        comparison_scope=("within matrix representation only",),
        required_caveats=(
            "Fragment-only software routes are not applicable without fragments.",
            "Keep author-peak and fixed-bin outputs in separate comparison scopes.",
        ),
        forbidden_interpretations=("Do not infer a universal method winner from matrix-only evidence.",),
        evidence_source=(
            "protocols/REPRESENTATION_FAIRNESS_PROTOCOL.md",
            "reports/G4_M4_TASK_SPECIFIC_RECOMMENDATION_POLICY_FREEZE_2026-07-17.md",
        ),
        claim_strength="strong_input_contract",
    ),
    RecommendationRule(
        rule_id="R02_fragments_available",
        scenario="Fragments are available for fragment-compatible software routes.",
        trigger=_has_fragments,
        recommended_methods=tuple(CORE_BASELINES + SOFTWARE_FRAGMENT_METHODS),
        methods_to_use_with_caution=(CICERO_SUBSET,),
        required_baselines=tuple(REPRESENTATION_BASELINES),
        required_negative_controls=tuple(NEGATIVE_CONTROLS),
        evaluation_tasks_to_run=("construction_qc", "method_portability", "runtime_failure_tracking"),
        comparison_scope=("fragment-compatible branch; do not pool with matrix-only branch without caveats",),
        required_caveats=(
            "Validate fragment reconstruction, genome annotation, barcode matching, and wrapper input contracts.",
            "ArchR-policy_B outputs are rank/correlation-safe; raw magnitude comparison is forbidden.",
            "Cicero is subset/scalability-only unless a full comparable run is explicitly validated.",
        ),
        forbidden_interpretations=(
            "Do not compare ArchR raw magnitudes directly.",
            "Do not describe Cicero subset output as full comparable.",
        ),
        evidence_source=(
            "reports/G3_ENVIRONMENT_REPRODUCIBILITY_2026-07-14.md",
            "reports/G4_M3A_ADMISSIBILITY_PORTABILITY_EVALUATION_2026-07-16.md",
        ),
        claim_strength="moderate_with_input_contract",
    ),
    RecommendationRule(
        rule_id="R03_high_sparsity_low_depth",
        scenario="High sparsity or low depth before method construction.",
        trigger=_high_sparsity_low_depth,
        recommended_methods=tuple(CORE_BASELINES),
        methods_to_use_with_caution=tuple(DISTAL_METHODS),
        required_baselines=tuple(REPRESENTATION_BASELINES),
        required_negative_controls=tuple(NEGATIVE_CONTROLS),
        evaluation_tasks_to_run=("construction_qc", "runtime_failure_tracking"),
        required_caveats=(
            "Prefer robust simple baselines under sparse or shallow inputs.",
            "Use distal-aware methods cautiously because mapping noise can dominate weak distal signal.",
            "Do not over-interpret RNA agreement under sparse/depth stress.",
        ),
        forbidden_interpretations=("Do not treat a high RNA proxy value as sufficient method evidence.",),
        evidence_source=(
            "reports/G4_M3C_REPLICATED_SIMULATION_TRUTH_RECOVERY_2026-07-16.md",
            "reports/G4_M3H_CROSS_STUDY_PAIRED_RNA_REPLICATION_2026-07-17.md",
        ),
        claim_strength="moderate_condition_specific",
    ),
    RecommendationRule(
        rule_id="R04_regulatory_or_distal_context",
        scenario="Regulatory interpretation is requested or high distal regulation is expected.",
        trigger=_regulatory_context,
        recommended_methods=("promoter_plus_distal_2kb_250kb_decay50kb_pc",),
        methods_to_use_with_caution=("distance_weighted_distal_250kb_decay50kb_pc", CICERO_SUBSET),
        required_baselines=("promoter_only_2kb_pc", "gene_body_pc", "raw_peak_lsi_baseline"),
        required_negative_controls=tuple(NEGATIVE_CONTROLS),
        evaluation_tasks_to_run=("marker_recovery", "tf_motif_consistency"),
        required_caveats=(
            "Distal-aware methods are regulatory-context analyses, not universal primary recommendations.",
            "TF/motif follow-up is recommended when resources allow, but pilot motif aggregation is not all-peak chromVAR.",
        ),
        forbidden_interpretations=("Do not state that distal-aware output is a universal winner.",),
        evidence_source=(
            "reports/G4_M3C_REPLICATED_SIMULATION_TRUTH_RECOVERY_2026-07-16.md",
            "reports/G4_M4_TASK_SPECIFIC_RECOMMENDATION_POLICY_FREEZE_2026-07-17.md",
        ),
        claim_strength="context_specific",
    ),
    RecommendationRule(
        rule_id="R05_rna_agreement_task",
        scenario="RNA agreement is the requested evaluation task.",
        trigger=_rna_task,
        recommended_methods=tuple(CORE_BASELINES),
        methods_to_use_with_caution=("accessibility_matched_random_peak_to_gene_control_pc",),
        required_baselines=tuple(REPRESENTATION_BASELINES),
        required_negative_controls=tuple(NEGATIVE_CONTROLS),
        evaluation_tasks_to_run=("rna_pseudobulk_agreement", "construction_qc"),
        required_caveats=(
            "RNA agreement is an evaluation proxy only, not construction tuning signal or ground truth.",
            "Adjacent RNA must be labeled as adjacent reference, not direct paired truth.",
            "Accessibility-matched random is required because RNA proxy metrics can be abundance-driven.",
        ),
        forbidden_interpretations=(
            "Do not select methods solely by RNA correlation.",
            "Do not treat adjacent RNA as direct paired truth.",
        ),
        evidence_source=(
            "reports/G4_M3H_CROSS_STUDY_PAIRED_RNA_REPLICATION_2026-07-17.md",
            "results/v1/g4/m4_recommendation_policy/frozen_task_specific_recommendation_policy.json",
        ),
        claim_strength="strong_policy_boundary",
    ),
    RecommendationRule(
        rule_id="R06_cross_dataset_benchmark",
        scenario="Cross-dataset benchmark or multi-representation comparison.",
        trigger=_cross_dataset,
        recommended_methods=tuple(CORE_BASELINES),
        methods_to_use_with_caution=tuple(SOFTWARE_FRAGMENT_METHODS + [CICERO_SUBSET]),
        required_baselines=tuple(REPRESENTATION_BASELINES),
        required_negative_controls=tuple(NEGATIVE_CONTROLS),
        evaluation_tasks_to_run=("construction_qc", "method_portability", "runtime_failure_tracking"),
        comparison_scope=(
            "within-representation comparison first",
            "portability/failure rate reported separately from performance metrics",
        ),
        required_caveats=(
            "Separate author-peak, fixed-bin, fragment-derived, and subset-only scopes.",
            "Do not pool software subset results with full comparable matrix results.",
        ),
        forbidden_interpretations=("Do not produce a final universal ranking across incompatible representations.",),
        evidence_source=(
            "protocols/REPRESENTATION_FAIRNESS_PROTOCOL.md",
            "results/v1/g4/m4_recommendation_policy/frozen_task_specific_recommendation_policy.json",
        ),
        claim_strength="strong_design_policy",
    ),
    RecommendationRule(
        rule_id="R07_simulation_truth_recovery",
        scenario="Simulation truth recovery is requested.",
        trigger=_simulation_truth,
        recommended_methods=tuple(CORE_BASELINES + DISTAL_METHODS),
        methods_to_use_with_caution=tuple(SOFTWARE_FRAGMENT_METHODS + [CICERO_SUBSET]),
        required_baselines=tuple(REPRESENTATION_BASELINES),
        required_negative_controls=tuple(NEGATIVE_CONTROLS),
        evaluation_tasks_to_run=(
            "true_ga_recovery",
            "promoter_distal_component_recovery",
            "peak_gene_link_recovery",
            "domain_recovery",
            "rna_vs_truth_proxy_analysis",
        ),
        comparison_scope=("mechanistic and empirical semi-synthetic validation contexts are reported separately",),
        required_caveats=(
            "Simulation truth can support controlled validation but must not be converted to a real-data final rank.",
            "Separate mechanistic and development-authorized empirical semi-synthetic evidence.",
        ),
        forbidden_interpretations=("Do not transfer simulation-derived ordering into real-data ordering.",),
        evidence_source=(
            "reports/G3_SIMULATION_READINESS_2026-07-11.md",
            "protocols/STATISTICAL_ANALYSIS_PLAN.md",
        ),
        claim_strength="simulation_only",
    ),
    RecommendationRule(
        rule_id="R08_software_portability",
        scenario="Software-standard workflow or portability is requested.",
        trigger=_software_portability,
        recommended_methods=tuple(SOFTWARE_FRAGMENT_METHODS),
        methods_to_use_with_caution=(CICERO_SUBSET,),
        required_baselines=tuple(REPRESENTATION_BASELINES),
        required_negative_controls=tuple(NEGATIVE_CONTROLS),
        evaluation_tasks_to_run=("method_portability", "runtime_failure_tracking"),
        comparison_scope=("software outputs are compared only where input contracts are satisfied",),
        required_caveats=(
            "Software methods require fragment/genome/API-compatible inputs.",
            "Failures are benchmark outputs and should be reported rather than hidden.",
        ),
        forbidden_interpretations=("Do not imply all software methods are runnable on every dataset.",),
        evidence_source=(
            "reports/G4_M3A_ADMISSIBILITY_PORTABILITY_EVALUATION_2026-07-16.md",
            "results/v1/g4/m4_recommendation_policy/frozen_task_specific_recommendation_policy.json",
        ),
        claim_strength="input_contract",
    ),
    RecommendationRule(
        rule_id="R09_missing_gene_annotation",
        scenario="Gene annotation is absent.",
        trigger=_missing_annotation,
        not_applicable_methods=tuple(CORE_BASELINES + DISTAL_METHODS + SOFTWARE_FRAGMENT_METHODS),
        required_baselines=tuple(REPRESENTATION_BASELINES),
        evaluation_tasks_to_run=("input_qc",),
        comparison_scope=("QC only until gene annotation is provided",),
        required_caveats=("Gene-annotation-dependent construction methods are not applicable without annotation.",),
        forbidden_interpretations=("Do not construct gene-level activity by labels, RNA, or clusters.",),
        evidence_source=("docs/input_formats.md", "registries/v1/methods.yaml"),
        claim_strength="strong_input_contract",
    ),
    RecommendationRule(
        rule_id="R10_missing_spatial_for_spatial_task",
        scenario="Spatial coordinates are absent for a spatial task.",
        trigger=_spatial_task_missing_spatial,
        not_applicable_methods=("spatial_domain_evaluation", "spatial_marker_map_evaluation"),
        evaluation_tasks_to_run=("construction_qc",),
        comparison_scope=("non-spatial construction QC only",),
        required_caveats=("Spatial clustering/domain tasks require spatial coordinates.",),
        forbidden_interpretations=("Do not report no-label spatial domain accuracy without spatial coordinates and labels.",),
        evidence_source=("registries/v1/tasks.yaml", "reports/G4_M3E_CROSS_STUDY_AUTHOR_DOMAIN_REPLICATION_2026-07-16.md"),
        claim_strength="strong_input_contract",
    ),
]


def rules_as_rows() -> list[dict[str, str]]:
    """Return a TSV-friendly rule provenance table."""

    rows = []
    for rule in RULES:
        rows.append(
            {
                "rule_id": rule.rule_id,
                "trigger_condition": rule.scenario,
                "recommended_action": "; ".join(rule.recommended_methods or rule.evaluation_tasks_to_run),
                "evidence_source": "; ".join(rule.evidence_source),
                "claim_strength": rule.claim_strength,
                "required_caveat": "; ".join(rule.required_caveats),
                "forbidden_interpretation": "; ".join(rule.forbidden_interpretations),
            }
        )
    return rows
