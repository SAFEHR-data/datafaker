"""Functions to install Python file references in ``config.yaml``."""
from collections.abc import Mapping, MutableMapping, Sequence
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from datafaker.utils import import_file, logger


def _make_where_from_annotation(
    query_def: Mapping[str, Any],
    fn_name: str,
    param_name: str,
) -> str:
    """Make a where clause from ``query`` value from the annotation."""
    if "where" not in query_def:
        return ""
    w = query_def["where"]
    if isinstance(w, str):
        return f" WHERE {w}"
    if isinstance(w, Sequence):
        return " WHERE " + " AND ".join(f'"({clause})"' for clause in w)
    logger.warning(
        '"where" in the query annotation of parameter "%s" of function "%s"'
        " needs to be a string or a list of strings",
        param_name,
        fn_name,
    )
    return ""


def _make_vars_from_annotation(
    query_def: Mapping[str, Any],
    fn_name: str,
    param_name: str,
) -> Mapping[str, Any]:
    """Make a variables dict from ``query`` value from the annotation."""
    if "vars" not in query_def:
        return {}
    vars_def = query_def["vars"]
    if isinstance(vars_def, Mapping):
        return vars_def
    if isinstance(vars_def, Sequence):
        return {v: v for v in query_def["vars"]}
    logger.warning(
        '"vars" in the query annotation of parameter "%s" of function "%s"'
        " needs to be a list of strings or a dict of strings to strings",
        param_name,
        fn_name,
    )
    return {}


def _add_count_vars_from_annotation(
    group_vars_out: MutableMapping[str, Any],
    query_def: Mapping[str, Any],
    fn_name: str,
    param_name: str,
) -> None:
    """Add ``GROUP BY`` clauses from ``count_vars``."""
    if "count_vars" not in query_def:
        return
    cntv = query_def["count_vars"]
    if isinstance(cntv, Mapping):
        group_vars_out.update({k: f"COUNT({v})" for k, v in cntv})
        return
    logger.warning(
        '"count_vars" needs to be a dict in the annotation for parameter %s of function %s',
        param_name,
        fn_name,
    )


def _add_ms_vars_from_annotation(
    group_vars_out: MutableMapping[str, Any],
    query_def: Mapping[str, Any],
    fn_name: str,
    param_name: str,
) -> None:
    """Add ``GROUP BY`` clauses from ``ms_vars``."""
    if "ms_vars" not in query_def:
        return
    msv = query_def["ms_vars"]
    if not isinstance(msv, Mapping):
        logger.warning(
            '"ms_vars" needs to be a dict in the annotation for parameter %s of function %s',
            param_name,
            fn_name,
        )
        return
    for k, v in msv.items():
        group_vars_out[k + "_count"] = f"COUNT({v})"
        group_vars_out[k + "_mean"] = f"AVG({v})"
        group_vars_out[k + "_stddev"] = f"STDDEV({v})"


def make_query_from_annotation(
    annotation_data: Any,
    fn_name: str,
    param_name: str,
) -> str | None:
    """
    Make new configuration items describing a query.

    The query's result will be passed as this parameter to this function.

    The annotation must be a dict with the following keys:

    ``comment``: A string describing the query in natural language.

    ``query``: Either a string containing the SQL query required, or
    a dict containing the following keys:

    * ``table``: The table to query. Could be "tablename AS alias" if you like.
    * ``vars`` (optional): Either a list of columns to extract from the table(s),
      or a dict of keys (the names of the keys in the dict to be passed to the
      annotated function) to values (the names of the columns to be extracted).
      At least one of ``vars``, ``ms_vars``, ``count_vars`` must be present.
    * ``where`` (optional): A SQL expression to filter the results.
    * ``count_vars`` (optional): A dict of keys to be passed to the function
      to values that are the names of the columns to be counted (could be
      ``*``; if the name of a column the result will be the number of non-null
      entries in that column). The query will be grouped by ``vars``.
    * ``ms_vars`` (optional): A dict of value names to columns to be analysed.
      The keys to be passed to the function will be name + ``_count`` for the
      number of non-null values in that column, name + ``_mean`` for the
      average value in that column and name + ``_stddev`` for the standard
      deviation of values in that column.

    :param annotation_data: The ``Annotation`` attached to the parameter.
    :param fn_name: The name of the function that the parameter is of.
    :param param_name: The name of the parameter with the annotation.
    :return: A mapping of new configuration items to add to the configuration,
    if the annotation had a well-defined query and comment value; otherwise
    an empty dict.
    """
    if not isinstance(annotation_data, Sequence):
        return None
    ann = annotation_data[0]
    if not isinstance(ann, Mapping) or "query" not in ann:
        return None
    if isinstance(ann["query"], str):
        return ann["query"]
    query_def = ann["query"]
    if "table" not in query_def:
        logger.warning(
            '"table" needs to be a key in the annotation for'
            ' the "query" value of parameter "%s" of function "%s"',
            param_name,
            fn_name,
        )
        return None
    table = query_def["table"]
    nongroup_vars = _make_vars_from_annotation(query_def, fn_name, param_name)
    where = _make_where_from_annotation(query_def, fn_name, param_name)
    group_vars: dict[str, Any] = {}
    _add_count_vars_from_annotation(group_vars, query_def, fn_name, param_name)
    _add_ms_vars_from_annotation(group_vars, query_def, fn_name, param_name)
    if group_vars and nongroup_vars:
        group_by = " GROUP BY " + ", ".join(f'"{v}"' for v in nongroup_vars)
    else:
        group_by = ""
    vars_exprs = ", ".join(
        f'{v} AS "{k}"' for k, v in {**nongroup_vars, **group_vars}.items()
    )
    return f"SELECT {vars_exprs} FROM {table}{group_by}{where}"


def _add_kwarg(
    kwargs_out: dict[str, Any], fn_name: str, param: Parameter
) -> list[dict[str, Any]]:
    """
    Add a kwargs configuration and return a ``src_stats`` query item.

    :param kwargs_out: The story generator's ``kwargs`` value to be updated.
    :param fn_name: The name of the story generator function.
    :param param: The parameter to specify.
    :return: A list of configuration items to add to the ``src_stats`` config, for
    all the queries this parameter requires.
    """
    if param.annotation is Parameter.empty:
        return []
    meta = param.annotation.__metadata__
    query = make_query_from_annotation(
        param.annotation.__metadata__, fn_name, param.name
    )
    if query is None:
        return []
    stat_name = f"story_auto__{fn_name}__{param.name}"
    if "comments" in meta[0]:
        comments = [meta[0]["comment"]]
    else:
        comments = []
    ssc = {
        "name": stat_name,
        "query": query,
        "comments": comments,
    }
    kwargs_out[param.name] = f'SRC_STATS["{stat_name}"]["results"]'
    return [ssc]


def install_stories_from(config: MutableMapping[str, Any], story_file: Path) -> bool:
    """
    Configure datafaker with the stories in a Python file.

    :param config: The contents of the configuration file, to be mutated.
    :param story_file: Path to the Python file containing the story generators.
    :return: True if the config was updated correctly, False if it was untouched
    because problems were encountered.
    """
    story_generators: list[Mapping[str, Any]] = []
    src_stats = [
        s
        for s in config.get("src_stats", [])
        if isinstance(s, Mapping)
        and "name" in s
        and not s["name"].startswith("story_auto__")
    ]
    story_module_name = story_file.stem
    story_module = import_file(story_file, story_module_name)
    for attr_name in dir(story_module):
        attr = getattr(story_module, attr_name)
        if (
            hasattr(attr, "__module__")
            and attr.__module__ == story_module_name
            and not attr_name.startswith("_")
            and callable(attr)
        ):
            kwargs: dict[str, None] = {}
            sig = signature(attr)
            for param in sig.parameters.values():
                src_stats += _add_kwarg(kwargs, attr, param)
            story_generators.append(
                {
                    "name": f"{story_module_name}.{attr_name}",
                    "num_stories_per_pass": 1,
                    "kwargs": kwargs,
                }
            )
    config["story_generators_module"] = story_module_name
    config["story_generators"] = story_generators
    config["src-stats"] = src_stats
    return True
