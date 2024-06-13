from pathlib import Path
from typing import Any

import yaml


def safe_update_dict_key(
    dict: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    """Update a dictionary with the given key-value pair.
    If the key does not exist, raise an error instead of adding it.
    """

    assert key in dict, key
    dict[key] = value


def safe_update_yaml(
    yaml_path: Path,
    *,
    updates: dict[str, Any] = {},
    injections: dict[str, Any] = {},
) -> None:
    print("updating yaml, path:", yaml_path)

    if not yaml_path.is_file():
        raise ValueError(f"yaml file not found: {yaml_path}")

    yaml_dict = yaml.safe_load(yaml_path.read_text())

    for k, v in updates.items():
        safe_update_dict_key(yaml_dict, k, v)

    for k, v in injections.items():
        yaml_dict[k] = v

    yaml_path.write_text(yaml.dump(yaml_dict))
