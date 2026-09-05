"""
Expansion of parameter specifications into concrete benchmark configurations.

infer_type            - coerce a CLI string into int, float, literal or str.
parse_cli_strings     - turn ["dim=1,2"] into {"dim": [1, 2]}.
build_cartesian_product - expand a parameter dict into one config per combination.
merge_config_sections - combine cartesian and explicit config sections.

A "configuration" throughout the framework is a plain dict of keyword arguments
passed to the benchmarked function, for example {"dim": 2, "algorithm": "sparsity"}.
"""

import ast
import itertools
from typing import Dict, List, Any


def infer_type(value: str) -> Any:
    """
    Coerce a string from the command line into a Python value.

    Tried in order: int, then float (only when the text looks like a float),
    then ast.literal_eval for values such as True or None. If none of these
    apply the original string is returned, so this never raises.

    Args:
        value: The raw string to coerce.

    Returns:
        The coerced value, or the unchanged string.
    """
    try:
        return int(value)
    except ValueError:
        pass

    try:
        if "." in value or "e" in value.lower():
            return float(value)
    except ValueError:
        pass

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    return value


def parse_cli_strings(param_args: List[str]) -> Dict[str, List[Any]]:
    """
    Parse list of strings "key=val1,val2" into dictionary.

    Example:
        Input: ["dim=1,2", "algo=A,B"]
        Output: {"dim": [1, 2], "algo": ["A", "B"]}
    """
    parsed = {}
    if not param_args:
        return parsed

    for arg in param_args:
        if "=" not in arg:
            continue

        key, val_str = arg.split("=", 1)

        raw_values = val_str.split(",")
        parsed[key] = [infer_type(v.strip()) for v in raw_values]

    return parsed


def build_cartesian_product(raw_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all possible configurations (Cartesian product) from parameters.

    Example:
        Input: {"dim": [1, 2], "algo": ["A", "B"]}
        Output:
            [
                {"dim": 1, "algo": "A"},
                {"dim": 1, "algo": "B"},
                {"dim": 2, "algo": "A"},
                {"dim": 2, "algo": "B"}
            ]
    """
    if not raw_params:
        return [{}]

    keys = list(raw_params.keys())
    values_list = [raw_params[k] for k in keys]

    configurations = []
    for combination in itertools.product(*values_list):
        config = dict(zip(keys, combination))
        configurations.append(config)

    return configurations


def merge_config_sections(
    cartesian_params: Dict[str, List[Any]] = None,
    explicit_configs: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Merge cartesian and explicit config sections into a unified list.

    This supports hybrid parameter configuration where users can:
    1. Auto-generate combinations via cartesian product
    2. Manually specify explicit configs to avoid invalid combinations

    Args:
        cartesian_params: Parameters for cartesian product generation
                         Example: {"dim": [1], "algorithm": ["graphic", "randomized"]}
        explicit_configs: List of explicit config dictionaries
                         Example: [{"dim": 2, "algorithm": "sparsity"}]

    Returns:
        Merged list of all configurations. A single empty config is returned
        when both sections are empty, so the caller always has one run to make.

    Example:
        Input: {"dim": [1], "algo": ["A", "B"]} and [{"dim": 2, "algo": "C"}]
        Output:
            [
                {"dim": 1, "algo": "A"},
                {"dim": 1, "algo": "B"},
                {"dim": 2, "algo": "C"}
            ]
    """
    configs = []

    # Generate cartesian product
    if cartesian_params:
        configs.extend(build_cartesian_product(cartesian_params))

    # Add explicit configs
    if explicit_configs:
        configs.extend(explicit_configs)

    return configs if configs else [{}]
