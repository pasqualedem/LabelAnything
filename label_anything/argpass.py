import argparse
from utils.utils import load_yaml
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    # Load the YAML configuration
    config_list = load_yaml('parameters.yaml')

    # Convert the list of dictionaries into a single dictionary
    config = {}
    for item in config_list:
        config.update(item)

     # Iterate through the config and add arguments to the parser
    for key, value in config.items():
        if isinstance(value, dict) and 'help' in value and 'value' in value:
            name = value['help']
            default_value = value['value']
        else:
            name = key.replace("_", " ").capitalize()
            default_value = value

        if isinstance(default_value, list):
            # Handle lists by specifying the type of the first element
            parser.add_argument(
                f"--{key}", type=type(default_value[0]), nargs="+", default=default_value, help=name)
        elif isinstance(default_value, dict):
            # Handle dictionaries by converting them to strings
            yaml_string = yaml.dump(default_value)
            parser.add_argument(f"--{key}", type=str,
                                default=yaml_string, help=name)
        else:
            parser.add_argument(
                f"--{key}", type=type(default_value), default=default_value, help=name)

    return parser.parse_args()
