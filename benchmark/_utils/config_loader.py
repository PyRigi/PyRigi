import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file with hybrid parameter support.

    Supports:
    - cartesian_params: Auto-generate combinations via cartesian product
    - explicit_configs: Manually specify exact parameter combinations

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration with 'configurations' key

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    from . import param_parser

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Extract hybrid configuration sections
    cartesian_params = config.get("cartesian_params", {})
    explicit_configs = config.get("explicit_configs", [])

    # Merge into unified configuration list
    config["configurations"] = param_parser.merge_config_sections(
        cartesian_params, explicit_configs
    )

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Target is required
    if "target" not in config:
        raise ValueError("Config must specify 'target' (function to benchmark)")

    # Dataset is required
    if "dataset" not in config:
        raise ValueError("Config must specify 'dataset' (path to graph files)")

    # At least one parameter section should be present
    has_cartesian = "cartesian_params" in config and config["cartesian_params"]
    has_explicit = "explicit_configs" in config and config["explicit_configs"]

    if not has_cartesian and not has_explicit:
        raise ValueError(
            "Config must specify at least one of: "
            "'cartesian_params' or 'explicit_configs'"
        )

    # Validate types
    if "cartesian_params" in config and not isinstance(
        config["cartesian_params"], dict
    ):
        raise ValueError("'cartesian_params' must be a dictionary")

    if "explicit_configs" in config and not isinstance(
        config["explicit_configs"], list
    ):
        raise ValueError("'explicit_configs' must be a list")


def merge_with_cli(config: Dict[str, Any], cli_args: Any) -> Dict[str, Any]:
    """
    Merge config file with CLI arguments. CLI takes precedence.

    Args:
        config: Configuration from YAML file
        cli_args: Parsed CLI arguments (argparse Namespace)

    Returns:
        Merged configuration dictionary
    """
    merged = config.copy()

    # CLI target overrides config
    if hasattr(cli_args, "target") and cli_args.target:
        merged["target"] = cli_args.target

    # CLI dataset overrides config
    if hasattr(cli_args, "dataset") and cli_args.dataset:
        merged["dataset"] = cli_args.dataset

    # CLI output overrides config
    if hasattr(cli_args, "output") and cli_args.output:
        merged["output"] = cli_args.output

    # CLI params override config parameters
    if hasattr(cli_args, "params") and cli_args.params:
        merged["cli_params"] = cli_args.params
    elif "parameters" in config:
        merged["cli_params"] = _convert_params_to_cli_format(config["parameters"])
    else:
        merged["cli_params"] = []

    # CLI min_rounds overrides config
    if hasattr(cli_args, "min_rounds") and cli_args.min_rounds is not None:
        merged["min_rounds"] = cli_args.min_rounds

    # CLI max_time overrides config
    if hasattr(cli_args, "max_time") and cli_args.max_time is not None:
        merged["max_time"] = cli_args.max_time

    # CLI warmup overrides config
    if hasattr(cli_args, "warmup") and cli_args.warmup is not None:
        merged["warmup"] = cli_args.warmup

    # CLI warmup_iterations overrides config
    if (
        hasattr(cli_args, "warmup_iterations")
        and cli_args.warmup_iterations is not None
    ):
        merged["warmup_iterations"] = cli_args.warmup_iterations

    return merged


def _convert_params_to_cli_format(params: Dict[str, List[Any]]) -> List[str]:
    """
    Convert config parameters dict to CLI string format.

    Args:
        params: Dict like {"dim": [1, 2], "algo": ["A", "B"]}

    Returns:
        List of strings like ["dim=1,2", "algo=A,B"]
    """
    cli_params = []
    for key, values in params.items():
        # Convert list to comma-separated string
        value_str = ",".join(str(v) for v in values)
        cli_params.append(f"{key}={value_str}")
    return cli_params
