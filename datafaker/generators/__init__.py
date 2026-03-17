"""Generators write generator function definitions and queries into config.yaml."""

from collections.abc import Mapping
from functools import lru_cache

from datafaker.generators.base import (
    ConstantGeneratorFactory,
    GeneratorFactory,
    MultiGeneratorFactory,
)
from datafaker.generators.choice import ChoiceGeneratorFactory
from datafaker.generators.continuous import (
    ContinuousDistributionGeneratorFactory,
    ContinuousLogDistributionGeneratorFactory,
    MultivariateLogNormalGeneratorFactory,
    MultivariateNormalGeneratorFactory,
)
from datafaker.generators.mimesis import (
    MimesisDateGeneratorFactory,
    MimesisDateTimeGeneratorFactory,
    MimesisFloatGeneratorFactory,
    MimesisIntegerGeneratorFactory,
    MimesisStringGeneratorFactory,
    MimesisTimeGeneratorFactory,
)
from datafaker.generators.partitioned import (
    NullPartitionedLogNormalGeneratorFactory,
    NullPartitionedNormalGeneratorFactory,
)


def everything_factory(config: Mapping) -> GeneratorFactory:
    """
    Get a factory that encapsulates all the other factories.

    :param config: The ``config.yaml`` configuration.
    """
    return MultiGeneratorFactory(
        MimesisStringGeneratorFactory(),
        MimesisIntegerGeneratorFactory(),
        MimesisFloatGeneratorFactory(),
        MimesisDateGeneratorFactory(),
        MimesisDateTimeGeneratorFactory(),
        MimesisTimeGeneratorFactory(),
        ContinuousDistributionGeneratorFactory(),
        ContinuousLogDistributionGeneratorFactory(),
        ChoiceGeneratorFactory(),
        ConstantGeneratorFactory(),
        MultivariateNormalGeneratorFactory(),
        MultivariateLogNormalGeneratorFactory(),
        NullPartitionedNormalGeneratorFactory(config),
        NullPartitionedLogNormalGeneratorFactory(config),
    )
