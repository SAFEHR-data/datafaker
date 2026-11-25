"""Build real/synthetic value distributions for statistical fidelity scoring."""

from abc import ABC, abstractmethod
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Float, cast, func, select

from datafaker.dialects import StdDev, random_sample
from datafaker.evaluators.feature_extractors import FeatureExtractor, IdentityExtractor

DEFAULT_SAMPLE_SIZE = 4000


@dataclass
class Distribution:
    """A discrete probability distribution over feature values."""

    probabilities: dict[Any, float]

    @property
    def vocabulary(self) -> set[Any]:
        """The set of feature values this distribution assigns mass to."""
        return set(self.probabilities.keys())

    def probability(self, key: Any) -> float:
        """Return the probability mass assigned to ``key`` (0.0 if unseen)."""
        return self.probabilities.get(key, 0.0)

    def as_vector(self, vocabulary: list[Any]) -> list[float]:
        """Render this distribution as a probability vector over ``vocabulary``."""
        return [self.probability(v) for v in vocabulary]


class DistributionBuilder(ABC):
    """Build a real or synthetic value's ``Distribution`` for one feature."""

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def __init__(
        self,
        engine,
        table,
        column,
        extractor: FeatureExtractor = IdentityExtractor(),
        sample_size=DEFAULT_SAMPLE_SIZE,
    ):
        """Initialize the engine/table/column/extractor/sample_size shared by every builder."""
        self.engine = engine
        self.table = table
        self.column = column
        self.extractor = extractor
        self.sample_size = sample_size

    @abstractmethod
    def build_from_table(self) -> Distribution:
        """Build the distribution of this feature over the real column."""

    @abstractmethod
    def build_from_values(self, values) -> Distribution:
        """Build the distribution of this feature over synthetic ``values``."""


# pylint: disable=too-many-instance-attributes
class HistogramBuilder(DistributionBuilder):
    """Bucket a continuous feature into a fixed-width histogram."""

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def __init__(
        self,
        engine,
        table,
        column,
        bins=10,
        extractor: FeatureExtractor = IdentityExtractor(),
        sample_size=DEFAULT_SAMPLE_SIZE,
    ):
        """Initialize a histogram builder for one column feature."""
        super().__init__(engine, table, column, extractor, sample_size)
        self.bins = bins

        # Learned from the real data
        self.mean = None
        self.stddev = None
        self.bottom = None
        self.width = None
        self.edges = None

    def build_from_table(self) -> Distribution:
        """Build the real column's histogram distribution."""
        feature_expr = self.extractor.expression(self.column)

        #
        # Mean and (sample) standard deviation, computed exactly over the
        # full table in one aggregate query - it returns a single row, so it
        # needs no sample_size cap. The feature is cast to a float first:
        # MSSQL's AVG of an integer expression is integer-truncated, and
        # Postgres' AVG of an integer returns NUMERIC (a Python Decimal).
        # AVG/STDDEV ignore NULLs, as the Python side ignores None features;
        # STDDEV (Postgres/DuckDB) and STDEV (MSSQL) are both the sample
        # standard deviation, like statistics.stdev, and are NULL for a
        # single value.
        #
        float_expr = cast(feature_expr, Float)
        with self.engine.connect() as conn:
            mean, stddev = conn.execute(
                select(func.avg(float_expr), StdDev(float_expr)).select_from(self.table)
            ).one()

        if mean is None:
            return Distribution({})

        self.mean = float(mean)
        self.stddev = 0.0 if stddev is None else float(stddev)

        width = 0.5 if self.stddev == 0 else self.stddev / 2
        self.edges = [
            float("-inf"),
            self.mean - 4 * width,
            self.mean - 3 * width,
            self.mean - 2 * width,
            self.mean - 1 * width,
            self.mean,
            self.mean + 1 * width,
            self.mean + 2 * width,
            self.mean + 3 * width,
            self.mean + 4 * width,
            float("inf"),
        ]
        bottom = self.mean - 4 * width

        if width == 0:
            width = 1.0

        #
        # SQL histogram, computed exactly over the full table (this is a
        # bounded, server-side aggregate query - it returns at most `bins`
        # rows - so, like the mean/stddev query above, it needs no
        # sample_size cap).
        #
        with self.engine.connect() as conn:
            # Compute the bucket expression once, in an inner subquery, and
            # group by the resulting materialized column in the outer query.
            # MSSQL rejects both `GROUP BY <alias>` (a SELECT-list alias
            # referenced from GROUP BY) and `GROUP BY <repeated expression>`
            # (its query planner does not recognize two separately-bound
            # occurrences of the same parameterized expression as
            # equivalent) - grouping by an actual column of a subquery is
            # the one form every dialect (Postgres/DuckDB/MSSQL) accepts.
            bucket_expr = func.floor((feature_expr - bottom) / width)
            inner = (
                select(bucket_expr.label("bucket")).select_from(self.table).subquery()
            )
            rows = conn.execute(
                select(
                    inner.c.bucket,
                    func.count().label("count"),  # pylint: disable=not-callable
                ).group_by(inner.c.bucket)
            ).all()

        # total must come from this same query, so the probabilities sum to 1.
        total = sum(count for bucket, count in rows if bucket is not None)

        if total == 0:
            return Distribution({})

        probs = {}

        for bucket, count in rows:
            if bucket is None:
                continue

            bucket = min(
                self.bins - 1,
                max(0, int(bucket) + 1),
            )

            probs[bucket] = probs.get(bucket, 0) + count / total

        return Distribution(probs)

    def build_from_values(self, values) -> Distribution:
        """Build the synthetic sample's histogram distribution."""
        if self.edges is None:
            # build_from_table() never learned edges - either it hasn't run
            # yet, or the real column had no usable values. Either way there
            # is nothing to bucket synthetic values against.
            return Distribution({})

        counts = Counter()
        total = 0

        for value in values:
            for feature in self.extractor.extract(value):
                feature = float(feature)
                bucket = bisect_right(self.edges, feature) - 1
                bucket = max(0, min(bucket, len(self.edges) - 2))
                counts[bucket] += 1
                total += 1

        if total == 0:
            return Distribution({})

        return Distribution({b: c / total for b, c in counts.items()})


class CategoryBuilder(DistributionBuilder):
    """Build a categorical distribution over a discrete feature's values."""

    def build_from_table(self):
        """
        Build the real column's categorical distribution.

        Counts are computed exactly over the full table with a SQL
        ``GROUP BY`` on the extractor's ``expression()``. Extractors with no
        SQL equivalent (``expression()`` raises ``NotImplementedError``,
        which includes every extractor that can yield several features per
        value, such as character bigrams) fall back to extracting features
        in Python from a random sample of ``sample_size`` rows.
        """
        try:
            feature_expr = self.extractor.expression(self.column)
        except NotImplementedError:
            counter = self._count_features_in_sample()
        else:
            counter = self._count_features_in_sql(feature_expr)

        total = sum(counter.values())
        if total == 0:
            return Distribution({})

        return Distribution({k: v / total for k, v in counter.items()})

    def _count_features_in_sql(self, feature_expr) -> Counter:
        """Count each non-NULL feature value over the full table."""
        with self.engine.connect() as conn:
            # Group by a column of a subquery rather than the expression
            # itself, as MSSQL requires (see HistogramBuilder above). This
            # returns one row per distinct feature value.
            inner = (
                select(feature_expr.label("feature")).select_from(self.table).subquery()
            )
            rows = conn.execute(
                select(
                    inner.c.feature,
                    func.count().label("count"),  # pylint: disable=not-callable
                )
                .where(inner.c.feature.is_not(None))
                .group_by(inner.c.feature)
            ).all()
        return Counter(dict(rows))

    def _count_features_in_sample(self) -> Counter:
        """Count features extracted in Python from a sample of the column."""
        counter: Counter = Counter()

        with self.engine.connect() as conn:
            # Bounded, randomly-ordered sample rather than the entire column,
            # so this scales to large tables (matches the synthetic sample
            # size for a fair, like-for-like comparison) without an
            # unordered LIMIT biasing the "real" distribution toward
            # whatever the DB returns first (often insertion order).
            rows = conn.execute(
                random_sample(
                    select(self.column).select_from(self.table),
                    self.sample_size,
                )
            )

            for (value,) in rows:
                if value is None:
                    continue
                for feature in self.extractor.extract(value):
                    if feature is None:
                        continue
                    counter[feature] += 1

        return counter

    def build_from_values(self, values):
        """Build the synthetic sample's categorical distribution."""
        counter = Counter()

        for value in values:
            if value is None:
                continue
            for feature in self.extractor.extract(value):
                if feature is None:
                    continue
                counter[feature] += 1

        total = sum(counter.values())
        if total == 0:
            # No features were extracted from the synthetic values; represent this
            # explicitly with a special missing token so metrics treat it as a
            # disjoint distribution (maximally different) compared to a real
            # distribution with concrete feature values.
            return Distribution({"__MISSING__": 1.0})

        return Distribution({k: v / total for k, v in counter.items()})
