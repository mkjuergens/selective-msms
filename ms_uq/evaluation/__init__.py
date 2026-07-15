from ms_uq.evaluation.rejection_curve import (
    rejection_curve,
    aurc_from_curve,
    aurc_trapezoid_from_curve,
    compute_oracle_aurc,
    compute_random_aurc,
)

from ms_uq.evaluation.metrics import (
    hit_at_k_ragged,
    compute_score_statistics,
    compute_correlations,
    compute_aurc_table,
    compute_fingerprint_losses,
    compute_all_losses,
    compute_aurc_all_losses,
    evaluate_uncertainty_vs_losses,
)

from ms_uq.evaluation.visualisation import (
    plot_aurc_bars,
    plot_risk_coverage_curves,
    plot_rc_and_aurc_paired,
    plot_member_vs_agg,
    plot_correlation_heatmap,
    plot_sgr_coverage_combined,
    plot_sgr_risk_calibration,
    DEFAULT_COLOR_MAP,
    CATEGORY_COLORS,
    AGGREGATION_COLORS,
    METRIC_LINESTYLES,
    get_metric_color,
    get_metric_category,
)

from ms_uq.evaluation.selective_risk import (
    SelectiveGuaranteedRisk,
    SGRResult,
    SGRComparisonResult,
    fit_sgr,
    compare_uncertainty_scores,
    sgr_risk_coverage_table,
    compute_binomial_bound,
    compute_hoeffding_bound,
    make_cal_eval_split,
    attach_eval_result,
)

__all__ = [
    # Rejection curves
    "rejection_curve",
    "aurc_from_curve",
    "aurc_trapezoid_from_curve",
    "compute_oracle_aurc",
    "compute_random_aurc",
    # Metrics
    "hit_at_k_ragged",
    "compute_score_statistics",
    "compute_correlations",
    "compute_aurc_table",
    "compute_fingerprint_losses",
    "compute_all_losses",
    "compute_aurc_all_losses",
    "evaluate_uncertainty_vs_losses",
    # Visualisation
    "plot_aurc_bars",
    "plot_risk_coverage_curves",
    "plot_rc_and_aurc_paired",
    "plot_member_vs_agg",
    "plot_correlation_heatmap",
    "plot_sgr_coverage_combined",
    "plot_sgr_risk_calibration",
    "DEFAULT_COLOR_MAP",
    "CATEGORY_COLORS",
    "AGGREGATION_COLORS",
    "METRIC_LINESTYLES",
    "get_metric_color",
    "get_metric_category",
    # Selective risk (SGR)
    "SelectiveGuaranteedRisk",
    "SGRResult",
    "SGRComparisonResult",
    "fit_sgr",
    "compare_uncertainty_scores",
    "sgr_risk_coverage_table",
    "compute_binomial_bound",
    "compute_hoeffding_bound",
    "make_cal_eval_split",
    "attach_eval_result",
]