"""Generators write generator function definitions and queries into config.yaml."""

from collections.abc import Mapping

from datafaker.proposers.base import (
    ConstantProposerFactory,
    MultiProposerFactory,
    ProposerFactory,
)
from datafaker.proposers.choice import ChoiceProposerFactory
from datafaker.proposers.continuous import (
    ContinuousDistributionProposerFactory,
    ContinuousLogDistributionProposerFactory,
    MultivariateLogNormalProposerFactory,
    MultivariateNormalProposerFactory,
)
from datafaker.proposers.mimesis import (
    MimesisDateProposerFactory,
    MimesisDateTimeProposerFactory,
    MimesisFloatProposerFactory,
    MimesisIntegerProposerFactory,
    MimesisStringProposerFactory,
    MimesisTimeProposerFactory,
)
from datafaker.proposers.partitioned import (
    NullPartitionedLogNormalProposerFactory,
    NullPartitionedNormalProposerFactory,
)


def everything_factory(config: Mapping) -> ProposerFactory:
    """
    Get a factory that encapsulates all the other factories.

    :param config: The ``config.yaml`` configuration.
    """
    return MultiProposerFactory(
        MimesisStringProposerFactory(),
        MimesisIntegerProposerFactory(),
        MimesisFloatProposerFactory(),
        MimesisDateProposerFactory(),
        MimesisDateTimeProposerFactory(),
        MimesisTimeProposerFactory(),
        ContinuousDistributionProposerFactory(),
        ContinuousLogDistributionProposerFactory(),
        ChoiceProposerFactory(),
        ConstantProposerFactory(),
        MultivariateNormalProposerFactory(),
        MultivariateLogNormalProposerFactory(),
        NullPartitionedNormalProposerFactory(config),
        NullPartitionedLogNormalProposerFactory(config),
    )
