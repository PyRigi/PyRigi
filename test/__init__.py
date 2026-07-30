def is_marker_selected(config, marker_name: str) -> bool:
    """Return if the given marker is selected."""
    expr = config.getoption("-m")
    if not expr:  # no filtering so everything selected
        return True
    return marker_name in expr and f"not {marker_name}" not in expr
