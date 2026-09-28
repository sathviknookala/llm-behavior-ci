"""Validation experiments.

Reports from this package separate methods that are implemented from
methods that have passed their required checks. A missing local runtime
or GPU reading stays missing.
"""

from llm_behavior_ci.experiments.validation import (
    AAContext,
    AADependenceReport,
    AAStudyRow,
    BaselineOutcome,
    CheckResult,
    HarmStudyReport,
    KLApproximationReport,
    KLPositionSample,
    ImplementedMethod,
    MethodSpec,
    MethodValidationStatus,
    ReferenceCase,
    StudyBudget,
    ValidationError,
    ValidationReport,
    aa_study_rows,
    apply_validation_reports,
    assess_harm_study,
    compare_plan_kl,
    harm_detection_power,
    implemented_methods,
    method_spec,
    minimum_checks,
    reference_library_status,
    validate_method,
)

__all__ = [
    "AAContext",
    "AADependenceReport",
    "AAStudyRow",
    "BaselineOutcome",
    "CheckResult",
    "HarmStudyReport",
    "KLApproximationReport",
    "KLPositionSample",
    "ImplementedMethod",
    "MethodSpec",
    "MethodValidationStatus",
    "ReferenceCase",
    "StudyBudget",
    "ValidationError",
    "ValidationReport",
    "aa_study_rows",
    "apply_validation_reports",
    "assess_harm_study",
    "compare_plan_kl",
    "harm_detection_power",
    "implemented_methods",
    "method_spec",
    "minimum_checks",
    "reference_library_status",
    "validate_method",
]
