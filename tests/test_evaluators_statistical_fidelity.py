"""Unit tests for datafaker.evaluators.statistical_fidelity."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, create_engine

from datafaker.evaluators.column_evaluator import EvaluationProfile
from datafaker.evaluators.statistical_fidelity import (
    EMAIL_PIPELINE,
    PROFILE_PIPELINES,
    StatisticalFidelity,
)
from tests.utils import DatafakerTestCase


class ProfilePipelinesTests(DatafakerTestCase):
    """Structural sanity checks for the built-in evaluation pipelines."""

    def test_every_evaluation_profile_has_a_pipeline(self) -> None:
        """Each EvaluationProfile enum member maps to a configured pipeline."""
        for profile in EvaluationProfile:
            self.assertIn(profile, PROFILE_PIPELINES)
            self.assertTrue(PROFILE_PIPELINES[profile])

    def test_pipeline_weights_sum_to_one(self) -> None:
        """Each profile's pipeline weights add up to 1.0, for a sane weighted average."""
        for profile, pipelines in PROFILE_PIPELINES.items():
            with self.subTest(profile=profile.name):
                total_weight = sum(p.weight for p in pipelines)
                self.assertAlmostEqual(1.0, total_weight)

    def test_pipeline_names_are_unique_within_a_profile(self) -> None:
        """Pipeline names double as dict keys in calculate_scores, so must be unique."""
        for profile, pipelines in PROFILE_PIPELINES.items():
            with self.subTest(profile=profile.name):
                names = [p.name for p in pipelines]
                self.assertEqual(len(names), len(set(names)))


def _make_table(columns: list[Column], rows: list[dict]) -> tuple:
    engine = create_engine("duckdb:///:memory:")
    metadata = MetaData()
    table = Table("data", metadata, *columns)
    metadata.create_all(engine)
    if rows:
        with engine.begin() as conn:
            conn.execute(table.insert(), rows)
    return engine, table


class StatisticalFidelityTests(DatafakerTestCase):
    """Test case for StatisticalFidelity."""

    def test_identical_real_and_synthetic_data_scores_perfectly(self) -> None:
        """A synthetic sample identical to the real data has zero divergence."""
        engine, table = _make_table(
            [Column("email", String)],
            [{"email": f"user{i}@example.com"} for i in range(30)],
        )
        fidelity = StatisticalFidelity(table.c.email, engine, sample_size=1000)
        fidelity.set_eval_pipelines(EvaluationProfile.EMAIL)

        synthetic = [f"user{i}@example.com" for i in range(30)]
        overall_score, pipeline_scores = fidelity.calculate_scores(synthetic)

        self.assertAlmostEqual(0.0, overall_score)
        self.assertEqual({p.name for p in EMAIL_PIPELINE}, set(pipeline_scores))
        for name, score in pipeline_scores.items():
            with self.subTest(pipeline=name):
                self.assertAlmostEqual(0.0, score)

    def test_completely_different_synthetic_data_scores_worse(self) -> None:
        """A synthetic sample sharing nothing with the real data scores worse."""
        engine, table = _make_table(
            [Column("category", String)],
            [{"category": "alpha"} for _ in range(30)],
        )
        fidelity = StatisticalFidelity(table.c.category, engine, sample_size=1000)
        fidelity.set_eval_pipelines(EvaluationProfile.CATEGORICAL)

        matching_score, _ = fidelity.calculate_scores(["alpha"] * 30)
        different_score, _ = fidelity.calculate_scores(["zzz_never_seen"] * 30)

        self.assertGreater(different_score, matching_score)

    def test_temporal_profile_scores_a_matching_date_column(self) -> None:
        """The TEMPORAL profile's two pipelines run cleanly against a date column."""
        engine, table = _make_table(
            [Column("ts", DateTime)],
            [{"ts": datetime(2020, 1, (i % 27) + 1)} for i in range(40)],
        )
        fidelity = StatisticalFidelity(table.c.ts, engine, sample_size=1000)
        fidelity.set_eval_pipelines(EvaluationProfile.TEMPORAL)

        synthetic = [datetime(2020, 1, (i % 27) + 1) for i in range(40)]
        overall_score, pipeline_scores = fidelity.calculate_scores(synthetic)

        self.assertAlmostEqual(0.0, overall_score)
        self.assertEqual({"timestamp", "day_of_week"}, set(pipeline_scores))

    def test_no_pipelines_set_scores_zero(self) -> None:
        """Without calling set_eval_pipelines, there's nothing to score."""
        engine, table = _make_table(
            [Column("n", Integer)], [{"n": i} for i in range(10)]
        )
        fidelity = StatisticalFidelity(table.c.n, engine, sample_size=1000)
        overall_score, pipeline_scores = fidelity.calculate_scores([1, 2, 3])
        self.assertEqual(0.0, overall_score)
        self.assertEqual({}, pipeline_scores)
