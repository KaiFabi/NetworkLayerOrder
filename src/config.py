import yaml


def load_config(file_path: str) -> dict:
    """Loads configuration file.

    Args:
        file_path: Path to yaml file.

    Returns:
        Dictionary holding content of yaml file.

    """
    with open(file_path, "r") as fp:
        try:
            return yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)
