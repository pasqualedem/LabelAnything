import argparse
from utils.utils import load_yaml
import yaml


import argparse
import yaml


# Helper function to determine argument type
def get_arg_type(value):
    if isinstance(value, int):
        return int
    elif isinstance(value, float):
        return float
    elif isinstance(value, str):
        return str
    elif isinstance(value, bool):
        return bool
    else:
        return type(value[0]) if isinstance(value, list) and value else str


# Helper function to parse the 'parameters' section
def parse_parameters(parameters):
    parsed_params = {}
    for key, value in parameters.items():
        if isinstance(value, dict) and "help" in value and "value" in value:
            name = value["help"]
            default_value = value["value"]
        else:
            name = key.replace("_", " ").capitalize()
            default_value = value

        arg_type = get_arg_type(default_value)

        if isinstance(default_value, list):
            # Handle lists by specifying the type of the first element
            parsed_params[key] = {
                "type": arg_type,
                "nargs": "+",
                "default": default_value,
                "help": name,
            }
        elif isinstance(default_value, dict):
            # Handle dictionaries by directly using the value
            parsed_params[key] = {
                "type": type(default_value),
                "default": default_value,
                "help": name,
            }
        else:
            parsed_params[key] = {
                "type": arg_type,
                "default": default_value,
                "help": name,
            }
    return parsed_params


def parse_args():
    parser = argparse.ArgumentParser()

    # Load the YAML configuration
    config = load_yaml("parameters.yaml")

    # Iterate through the config and add arguments to the parser
    for section_name, section_data in config.items():
        if section_name == "parameters":
            parameters = parse_parameters(section_data)
            for key, value in parameters.items():
                parser.add_argument(f"--{key}", **value)
        else:
            for key, value in section_data.items():
                if "help" in value and "value" in value:
                    name = value["help"]
                    default_value = value["value"]
                else:
                    name = key.replace("_", " ").capitalize()
                    default_value = value

                arg_type = get_arg_type(default_value)

                if isinstance(default_value, list):
                    # Handle lists by specifying the type of the first element
                    parser.add_argument(
                        f"--{key}",
                        type=arg_type,
                        nargs="+",
                        default=default_value,
                        help=name,
                    )
                elif isinstance(default_value, dict):
                    # Handle dictionaries by directly using the value
                    parser.add_argument(
                        f"--{key}",
                        type=type(default_value),
                        default=default_value,
                        help=name,
                    )
                else:
                    parser.add_argument(
                        f"--{key}", type=arg_type, default=default_value, help=name
                    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.train_params["seed"]["value"])
