"""Story generators which describe their own queries and can therefore be installed."""
from collections.abc import Iterable
from typing import Annotated, Any

def string_story_one_sd(
    stats: Annotated[dict, {
        "query": {
            "ms_vars": {"freq": "frequency"},
            "table": "string",
        },
        "comment": "Frequency mean and standard deviation",
    }],
) -> Iterable[tuple[str, dict[str, Any]]]:
    man = yield("manufacturer", {"name": "one"})
    mod = yield ("model", {
        "name": "one_sd",
        "manufacturer_id": man["id"]
    })
    yield("string", {
        "model_id": mod["id"],
        "position": 0,
        "frequency": stats[0]["freq_mean"] - stats[0]["freq_stddev"],
    })
    yield("string", {
        "model_id": mod["id"],
        "position": stats[0]["freq_count"],
        "frequency": stats[0]["freq_mean"] + stats[0]["freq_stddev"],
    })
