from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ms_uq.evaluation.revision_features import (
    retrieval_temperature_features,
    score_cache_name,
    score_gap_at_k,
    spearman_confidence_log_candidates,
)
from ms_uq.evaluation.selective_risk import make_cal_eval_split
from ms_uq.data import candidate_fps_to_dense
from ms_uq.utils import resolve_candidate_paths


def test_sgr_split_is_deterministic_and_disjoint():
    cal1, eval1 = make_cal_eval_split(101, cal_fraction=0.5, seed=42)
    cal2, eval2 = make_cal_eval_split(101, cal_fraction=0.5, seed=42)
    assert np.array_equal(cal1, cal2)
    assert np.array_equal(eval1, eval2)
    assert set(cal1).isdisjoint(set(eval1))
    assert sorted(np.concatenate([cal1, eval1]).tolist()) == list(range(101))


def test_sgr_exports_all_manuscript_single_score_curves(tmp_path: Path):
    from ms_uq.evaluation.revision_reporting import SGR_SINGLE_MEASURES, run_sgr_stability

    rows = []
    for run_label in ["mlp_formula", "transformer_formula"]:
        for split in ["val", "test"]:
            for k in [1, 5, 20]:
                for index in range(20):
                    confidence = 1.0 - index / 25.0
                    rows.append({
                        "run_label": run_label,
                        "split": split,
                        "evaluation_candidate_setting": "formula_official_capped",
                        "K": k,
                        "query_id": f"{run_label}-{split}-{index}",
                        "molecule_group_id": f"mol-{index}",
                        "hit": float(index % 3 != 0),
                        "aurc_convention": "manuscript_trapezoid_seed42",
                        "candidate_tie_break": "source_order",
                        "confidence": confidence,
                        "score_gap": confidence ** 2,
                        "retrieval_total": confidence * 0.95,
                        "retrieval_aleatoric": confidence * 0.9,
                        "retrieval_epistemic": confidence * 0.8,
                        "rank_var_1": confidence * 0.7,
                        "rank_var_5": confidence * 0.6,
                        "rank_var_20": confidence * 0.5,
                    })
    run_sgr_stability(
        pd.DataFrame(rows), tmp_path, seeds=[42], target_risks=[0.5], delta=0.1,
    )
    result = pd.read_csv(tmp_path / "sgr_evaluation.csv")
    observed = result.groupby(["run_label", "K"])["measure"].agg(set)
    assert all(set(SGR_SINGLE_MEASURES) <= measures for measures in observed)
    from scripts.plot_sgr_analysis import plot_sgr_coverage

    plot_input = result.assign(
        loss=result["K"].map(lambda k: f"hit@{k}"),
        category="retrieval",
    )
    with pytest.raises(ValueError, match="multiple run labels"):
        plot_sgr_coverage(plot_input, tmp_path / "mixed.pdf")


def test_normalized_entropy_bounds():
    scores_stack = torch.tensor([[2.0, 1.0, 0.0, 0.0], [1.8, 1.1, 0.1, -0.2]])
    scores_flat = scores_stack.mean(dim=0)
    ptr = torch.tensor([0, 2, 4])
    feats = retrieval_temperature_features(scores_stack, scores_flat, ptr, temperature=1.0, top_ks=[1])
    h = feats["normalized_entropy"]
    assert h.shape == (2,)
    assert np.all(h >= -1e-6)
    assert np.all(h <= 1.0 + 1e-6)


def test_score_gap_at_k_imputes_short_candidate_lists_as_high_confidence():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    ptr = torch.tensor([0, 2, 5])
    values, imp = score_gap_at_k(scores, ptr, k=2)
    assert imp.n_imputed == 1
    assert np.isfinite(values).all()
    assert values[0] > values[1]


def test_probability_score_cache_name_contains_temperature():
    assert score_cache_name("ranker", "score", 0.003) == "scores_ranker_score.pt"
    assert score_cache_name("ranker", "probability", 0.003) == "scores_ranker_probability_T0p003.pt"


def test_confidence_candidate_size_spearman():
    rho, p = spearman_confidence_log_candidates(
        np.asarray([0.9, 0.8, 0.2, 0.1]),
        np.log(np.asarray([1, 2, 10, 20])),
    )
    assert rho < 0
    assert 0 <= p <= 1


def test_candidate_fps_to_dense_accepts_dense_and_packed():
    dense = np.asarray(
        [
            [1, 0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 0],
        ],
        dtype=bool,
    )
    assert np.array_equal(candidate_fps_to_dense(dense, n_candidates=2, fp_size=8), dense)

    packed = np.packbits(dense.reshape(-1).astype(np.uint8), bitorder="big")
    unpacked = candidate_fps_to_dense(packed, n_candidates=2, fp_size=8)
    assert np.array_equal(unpacked, dense)


def test_build_candidate_json_from_npz_tiny(tmp_path: Path):
    from scripts.build_candidate_json_from_npz import build_compat_candidate_json

    inchi_path = tmp_path / "MassSpecGym_retrieval_candidates_mass_inchi.npz"
    np.savez(
        inchi_path,
        CCO=np.asarray(["A", "B"]),
        CCN=np.asarray(["C"]),
    )
    out_path = tmp_path / "MassSpecGym_retrieval_candidates_mass.json"
    candidate_map = build_compat_candidate_json(inchi_path, out_path)

    assert out_path.exists()
    assert list(candidate_map) == ["CCO", "CCN"]
    assert [len(v) for v in candidate_map.values()] == [2, 1]
    with pytest.raises(FileExistsError):
        build_compat_candidate_json(inchi_path, out_path)


def test_formula_uncapped_path_resolution(tmp_path: Path):
    names = [
        "MassSpecGym_retrieval_candidates_formula_uncapped.json",
        "MassSpecGym_retrieval_candidates_formula_uncapped_fps.npz",
        "MassSpecGym_retrieval_candidates_formula_uncapped_inchi.npz",
    ]
    for name in names:
        (tmp_path / name).touch()
    resolved = resolve_candidate_paths(tmp_path, "formula_uncapped")
    assert [p.name for p in resolved] == names


def test_build_candidate_helpers_tiny_json(tmp_path: Path):
    pytest.importorskip("rdkit")
    from scripts.build_candidate_helpers import build_candidate_arrays, normalize_candidate_map, output_paths

    raw = {"CCO": ["CCO", "CC"]}
    cmap = normalize_candidate_map(raw)
    fps, inchis = build_candidate_arrays(cmap, fp_size=128)
    assert fps["CCO"].shape == (2, 128)
    assert inchis["CCO"].shape == (2,)
    fp_path, inchi_path = output_paths(tmp_path / "MassSpecGym_retrieval_candidates_formula_uncapped.json")
    assert fp_path.name == "MassSpecGym_retrieval_candidates_formula_uncapped_fps.npz"
    assert inchi_path.name == "MassSpecGym_retrieval_candidates_formula_uncapped_inchi.npz"


def test_prepare_uncapped_helpers_rekeys_packs_and_preserves_order(tmp_path: Path):
    pytest.importorskip("rdkit")
    from scripts.prepare_uncapped_formula_test import (
        build_shards,
        consolidate_shards,
        inchikey_2d,
        resolve_query_candidates,
    )

    original_query = "C(C)O"
    canonical_query = "CCO"
    source = {canonical_query: [canonical_query, "OCC"]}
    metadata = pd.DataFrame([{
        "smiles": original_query,
        "inchikey": inchikey_2d(original_query),
        "formula": "C2H6O",
        "fold": "test",
    }])
    candidate_map, query_meta, methods = resolve_query_candidates(source, metadata)
    assert list(candidate_map) == [original_query]
    assert methods["canonical"] == 1

    shard_dir = tmp_path / "shards"
    results = build_shards(candidate_map, query_meta, shard_dir, fp_size=128, workers=1)
    assert results == [(0, 2, 2, 0)]

    fp_path = tmp_path / "fps.npz"
    inchi_path = tmp_path / "inchi.npz"
    consolidate_shards(candidate_map, shard_dir, fp_path, inchi_path)
    with np.load(fp_path) as fps, np.load(inchi_path) as inchis:
        dense = candidate_fps_to_dense(fps[original_query], n_candidates=2, fp_size=128)
        assert dense.shape == (2, 128)
        assert inchis[original_query].tolist() == [
            inchikey_2d(canonical_query).encode(),
            inchikey_2d("OCC").encode(),
        ]


def test_biencoder_cosine_matrix_matches_pairwise_cosine():
    from ms_uq.inference.retrieve import BiencoderRanker

    rng = torch.Generator().manual_seed(7)
    queries = torch.rand((3, 32), generator=rng)
    candidates = torch.rand((11, 32), generator=rng)
    ranker = BiencoderRanker(n_bits=32, sim_func="cossim")
    actual = ranker(queries, candidates)
    expected = torch.nn.functional.cosine_similarity(
        queries[:, None, :].expand(-1, candidates.shape[0], -1),
        candidates[None, :, :].expand(queries.shape[0], -1, -1),
        dim=-1,
    )
    assert actual.shape == (3, 11)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_meta_training_grouped_cv_smoke():
    from scripts.run_meta_score_analysis import train_meta_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + 0.2 * rng.normal(size=60) > 0).astype(int)
    groups = np.repeat(np.arange(20), 3)
    model, cv, best_c = train_meta_model(X, y, groups, [0.1, 1.0])
    pred = model.predict_proba(X[:5])[:, 1]
    assert pred.shape == (5,)
    assert best_c in {0.1, 1.0}
    assert not cv.empty


def test_candidate_control_is_deterministic_deduplicated_and_target_protected():
    from ms_uq.evaluation.revision_candidates import select_candidate_indices

    candidates = ["NEGB", "TARGET", "NEGA", "NEGB"]
    first = select_candidate_indices(candidates, ["TARGET"], cap=3, seed=42)
    second = select_candidate_indices(list(reversed(candidates)), ["TARGET"], cap=3, seed=42)
    assert set(first.candidate_ids) == set(second.candidate_ids)
    assert "TARGET" in first.candidate_ids
    assert len(first.candidate_ids) == len(set(first.candidate_ids)) == 3
    assert first.duplicate_occurrences == 1


def test_canonical_candidate_view_is_input_order_independent():
    from ms_uq.evaluation.revision_candidates import canonical_candidate_view

    ids = np.asarray(["B", "A", "B"])
    fps = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
    canonical_ids, canonical_fps, _ = canonical_candidate_view(ids, fps, fp_size=4)
    perm = np.asarray([2, 0, 1])
    shuffled_ids, shuffled_fps, _ = canonical_candidate_view(ids[perm], fps[perm], fp_size=4)
    assert canonical_ids.tolist() == shuffled_ids.tolist() == ["A", "B"]
    assert np.array_equal(canonical_fps, shuffled_fps)


def test_discrete_aurc_and_analytic_random_baseline():
    from ms_uq.evaluation.revision_features import canonical_aurc_table, discrete_aurc

    loss = np.asarray([0.0, 1.0])
    confidence = np.asarray([1.0, 0.0])
    assert discrete_aurc(confidence, loss, query_ids=["a", "b"]) == pytest.approx(0.25)
    aurc, rel = canonical_aurc_table({"score": confidence}, {"hit@1": 1.0 - loss}, ["a", "b"])
    assert aurc.loc["random", "hit@1"] == pytest.approx(loss.mean())
    assert aurc.loc["oracle", "hit@1"] == pytest.approx(0.25)
    assert rel.loc["score", "hit@1"] == pytest.approx(0.0)


def test_canonical_temperature_features_are_confidence_oriented():
    from ms_uq.evaluation.revision_features import softmax_temperature_features

    stack = torch.tensor([[0.9, 0.2, 0.1], [0.8, 0.3, 0.0]], dtype=torch.float64)
    aggregate = stack.mean(dim=0)
    features = softmax_temperature_features(stack, aggregate, torch.tensor([0, 3]), 0.3, top_ks=[1, 2])
    assert features["confidence"][0] > 1 / 3
    assert 0 <= features["normalized_entropy"][0] <= 1
    assert features["retrieval_total"][0] <= 0
    assert features["retrieval_aleatoric"][0] <= 0
    assert features["retrieval_epistemic"][0] == pytest.approx(
        features["retrieval_total"][0] - features["retrieval_aleatoric"][0]
    )
    assert features["rank_var_1"][0] <= 0


def test_meta_pipeline_has_no_imputer_and_uses_strict_logistic_settings():
    from scripts.run_meta_score_analysis import make_pipeline

    pipeline = make_pipeline(meta_model="logistic")
    assert list(pipeline.named_steps) == ["scaler", "logreg"]
    assert pipeline.named_steps["logreg"].max_iter == 10000
    assert pipeline.named_steps["logreg"].tol == pytest.approx(1e-8)


def test_constant_loss_has_defined_aurc_and_undefined_relative_aurc():
    from ms_uq.evaluation.revision_features import canonical_aurc_table

    confidence = np.asarray([0.8, 0.2])
    aurc, rel = canonical_aurc_table(
        {"score": confidence}, {"hit@1": np.ones(2)}, ["a", "b"]
    )
    assert aurc.loc["score", "hit@1"] == pytest.approx(0.0)
    assert np.isnan(rel.loc["score", "hit@1"])



def test_record_candidate_cap_preserves_duplicate_occurrences_and_targets():
    from ms_uq.evaluation.revision_candidates import select_record_candidate_indices

    records = ["dup", "target", "dup", "negative-3", "negative-4"]
    identities = ["D", "T", "D", "N3", "N4"]
    first = select_record_candidate_indices(records, identities, ["T"], cap=4, seed=42)
    second = select_record_candidate_indices(records, identities, ["T"], cap=4, seed=42)
    assert first == second
    assert len(first.source_indices) == 4
    assert tuple(sorted(first.source_indices)) == first.source_indices
    assert 1 in first.source_indices
    assert first.target_indices == (1,)
    assert first.n_exact_duplicate_occurrences == 1
    assert first.n_connectivity_duplicate_occurrences == 1


def test_record_candidate_cap_keeps_all_records_when_pool_is_under_cap():
    from ms_uq.evaluation.revision_candidates import select_record_candidate_indices

    result = select_record_candidate_indices(
        ["same", "same", "target"], ["D", "D", "T"], ["T"], cap=256, seed=42
    )
    assert result.source_indices == (0, 1, 2)


def test_preserve_score_bundle_keeps_order_duplicates_and_float32(tmp_path: Path):
    from scripts.canonicalize_score_bundle import preserve_score_bundle

    metadata = pd.DataFrame([{
        "identifier": "q1", "smiles": "query", "inchikey": "TARGET-AA", "fold": "test",
    }])
    dataset_tsv = tmp_path / "MassSpecGym.tsv"
    metadata.to_csv(dataset_tsv, sep="\t", index=False)
    (tmp_path / "MassSpecGym_retrieval_candidates_formula.json").write_text(
        '{"query": ["a", "target", "a"]}'
    )
    np.savez(
        tmp_path / "MassSpecGym_retrieval_candidates_formula_fps.npz",
        query=np.asarray([[1, 0], [0, 1], [1, 0]], dtype=bool),
    )
    np.savez(
        tmp_path / "MassSpecGym_retrieval_candidates_formula_inchi.npz",
        query=np.asarray(["A", "TARGET", "A"]),
    )
    raw_path = tmp_path / "raw.pt"
    output_path = tmp_path / "records.pt"
    raw = {
        "scores_flat": torch.tensor([0.2, 0.9, 0.2], dtype=torch.float32),
        "scores_stack_flat": torch.tensor([[0.2, 0.9, 0.2]], dtype=torch.float32),
        "labels_flat": torch.tensor([0.0, 1.0, 0.0]),
        "ptr": torch.tensor([0, 3]),
    }
    torch.save(raw, raw_path)
    summary = preserve_score_bundle(
        raw_path, output_path, dataset_tsv, tmp_path, "formula", "test",
        "fingerprint", "precomputed",
    )
    preserved = torch.load(output_path, map_location="cpu")
    assert preserved["candidate_record_policy"] == "preserve"
    assert preserved["candidate_deduplication"] == "none"
    assert preserved["scores_flat"].dtype == torch.float32
    assert torch.equal(preserved["scores_flat"], raw["scores_flat"])
    assert torch.equal(preserved["labels_flat"], raw["labels_flat"])
    assert preserved["record_candidate_counts"].tolist() == [3]
    assert summary["n_raw_scores"] == summary["n_record_scores"] == 3


def test_preserve_score_bundle_can_materialize_prefix(tmp_path: Path):
    from scripts.canonicalize_score_bundle import preserve_score_bundle

    metadata = pd.DataFrame([
        {"identifier": "q1", "smiles": "query", "inchikey": "TARGET-AA", "fold": "test"},
        {"identifier": "q2", "smiles": "query", "inchikey": "TARGET-AA", "fold": "test"},
    ])
    dataset_tsv = tmp_path / "MassSpecGym.tsv"
    metadata.to_csv(dataset_tsv, sep="\t", index=False)
    (tmp_path / "MassSpecGym_retrieval_candidates_formula.json").write_text(
        '{"query": ["a", "target", "a"]}'
    )
    np.savez(
        tmp_path / "MassSpecGym_retrieval_candidates_formula_fps.npz",
        query=np.asarray([[1, 0], [0, 1], [1, 0]], dtype=bool),
    )
    np.savez(
        tmp_path / "MassSpecGym_retrieval_candidates_formula_inchi.npz",
        query=np.asarray(["A", "TARGET", "A"]),
    )
    raw_path = tmp_path / "raw.pt"
    output_path = tmp_path / "prefix.pt"
    raw = {
        "scores_flat": torch.tensor([0.2, 0.9, 0.2, 0.3, 0.8, 0.3]),
        "scores_stack_flat": torch.tensor([
            [0.2, 0.9, 0.2, 0.3, 0.8, 0.3],
            [0.1, 1.0, 0.1, 0.2, 0.9, 0.2],
        ]),
        "labels_flat": torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        "ptr": torch.tensor([0, 3, 6]),
    }
    torch.save(raw, raw_path)
    summary = preserve_score_bundle(
        raw_path, output_path, dataset_tsv, tmp_path, "formula", "test",
        "fingerprint", "precomputed", max_queries=1,
    )
    prefix = torch.load(output_path, map_location="cpu")
    assert prefix["ptr"].tolist() == [0, 3]
    assert tuple(prefix["scores_stack_flat"].shape) == (2, 3)
    assert prefix["scores_stack_flat"].untyped_storage().nbytes() == (
        prefix["scores_stack_flat"].numel() * prefix["scores_stack_flat"].element_size()
    )
    assert prefix["query_ids"] == ["q1"]
    assert summary["n_queries"] == 1


def test_chunked_bitwise_uncertainties_match_full_computation():
    from ms_uq.unc_measures.eval_measures import compute_fingerprint_uncertainties

    generator = torch.Generator().manual_seed(42)
    predictions = torch.rand((7, 5, 16), generator=generator)
    measures = ["bitwise_total", "bitwise_aleatoric", "bitwise_epistemic"]
    full = compute_fingerprint_uncertainties(predictions, measures)
    chunked = compute_fingerprint_uncertainties(predictions.half(), measures, batch_size=3)
    for measure in measures:
        assert np.allclose(full[measure], chunked[measure], atol=2e-3, rtol=2e-3)


def test_paper_figures_exclude_normalized_entropy_and_include_bitwise_aleatoric():
    from scripts.plot_meta_joint_results import MANUSCRIPT_MEASURES
    from scripts.run_temperature_sensitivity import _metric_order

    assert "normalized_entropy" not in MANUSCRIPT_MEASURES
    assert "normalized_entropy" not in _metric_order(5)
    assert "bitwise_aleatoric" in MANUSCRIPT_MEASURES


def test_manuscript_aurc_uses_trapezoid_and_seed42_random_baseline():
    from ms_uq.evaluation.revision_features import canonical_aurc_table

    confidence = np.asarray([0.9, 0.6, 0.3, 0.1])
    hit = np.asarray([1.0, 0.0, 1.0, 0.0])
    manuscript, manuscript_rel = canonical_aurc_table(
        {"score": confidence}, {"hit@1": hit},
        convention="manuscript_trapezoid_seed42", tie_break="source_order",
    )
    discrete, _ = canonical_aurc_table(
        {"score": confidence}, {"hit@1": hit},
        convention="discrete_prefix_mean", tie_break="source_order",
    )
    assert manuscript.loc["score", "hit@1"] != pytest.approx(discrete.loc["score", "hit@1"])
    assert manuscript.loc["random", "hit@1"] != pytest.approx((1.0 - hit).mean())
    assert np.isfinite(manuscript_rel.loc["score", "hit@1"])


def test_record_formula_cap_builder_preserves_selected_source_order(tmp_path: Path):
    import json
    from ms_uq.evaluation.revision_candidates import build_record_preserving_formula_cap

    metadata = pd.DataFrame([{
        "identifier": "q1", "smiles": "query", "inchikey": "TARGET-AA", "fold": "test",
    }])
    dataset_tsv = tmp_path / "MassSpecGym.tsv"
    metadata.to_csv(dataset_tsv, sep="\t", index=False)
    np.save(tmp_path / "inchis.npy", np.asarray(["TARGET-PRECOMPUTED"]))
    records = ["dup", "target", "dup", "negative-3", "negative-4"]
    (tmp_path / "source.json").write_text(json.dumps({"query": records}))
    fps = np.asarray([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=bool)
    identities = np.asarray(["D", "TARGET", "D", "N3", "N4"])
    np.savez(tmp_path / "source_fps.npz", query=fps)
    np.savez(tmp_path / "source_inchi.npz", query=identities)
    summary = build_record_preserving_formula_cap(
        dataset_tsv, tmp_path / "source.json", tmp_path / "source_fps.npz",
        tmp_path / "source_inchi.npz", tmp_path / "inchis.npy", tmp_path,
        cap=4, seed=42, fp_size=8,
    )
    prefix = tmp_path / "MassSpecGym_retrieval_candidates_formula_pubchem_record_capped256"
    selected_records = json.loads(prefix.with_suffix(".json").read_text())["query"]
    with np.load(prefix.with_name(prefix.name + "_inchi.npz")) as selected_inchis:
        selected_ids = selected_inchis["query"].astype(str).tolist()
    from ms_uq.evaluation.revision_candidates import select_record_candidate_indices
    expected = select_record_candidate_indices(records, identities, ["TARGET"], cap=4, seed=42)
    assert selected_records == [records[index] for index in expected.source_indices]
    assert selected_ids == [identities[index] for index in expected.source_indices]
    assert "target" in selected_records
    assert len(selected_records) == len(selected_ids) == 4
    assert summary.loc[0, "n_target_occurrences_capped"] == 1
    assert bool(summary.loc[0, "source_order_preserved"])


def test_revision_runner_expands_downstream_stage_dependencies():
    from scripts.run_revision_rerun import STAGES, expand_stage_dependencies

    assert expand_stage_dependencies(["metrics"]) == [
        "preflight", "candidates", "scores", "metrics",
    ]
    assert expand_stage_dependencies(["report", "validate"]) == STAGES


def test_canonical_temperature_matches_rankwise_training_temperature():
    import inspect

    import yaml

    from ms_uq.evaluation.metrics import compute_score_statistics
    from ms_uq.inference.retrieve import ragged_softmax, scores_from_loader
    from ms_uq.unc_measures.eval_measures import compute_uncertainties
    from ms_uq.unc_measures.retrieval_unc import RetrievalUncertainty
    from scripts.run_evaluation import EvalConfig
    from scripts.run_revision_rerun import EVALUATION_TEMPERATURE
    from scripts.run_sgr_evaluation import SGRConfig

    expected = 0.003
    assert EVALUATION_TEMPERATURE == pytest.approx(expected)
    assert EvalConfig().temperature == pytest.approx(expected)
    assert SGRConfig().temperature == pytest.approx(expected)
    assert inspect.signature(compute_score_statistics).parameters["temperature"].default == pytest.approx(expected)
    assert inspect.signature(scores_from_loader).parameters["temperature"].default == pytest.approx(expected)
    assert inspect.signature(ragged_softmax).parameters["temperature"].default == pytest.approx(expected)
    assert inspect.signature(compute_uncertainties).parameters["temperature"].default == pytest.approx(expected)
    assert inspect.signature(RetrievalUncertainty).parameters["temperature"].default == pytest.approx(expected)

    def configured_temperatures(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "temperature":
                    yield float(item)
                yield from configured_temperatures(item)
        elif isinstance(value, list):
            for item in value:
                yield from configured_temperatures(item)

    config_dir = Path(__file__).resolve().parents[1] / "config"
    for config_path in config_dir.glob("*.yml"):
        values = list(configured_temperatures(yaml.safe_load(config_path.read_text())))
