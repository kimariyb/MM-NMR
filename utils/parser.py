import argparse
import os
import yaml


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


def number(text):
    r"""
    Converts a string to a number.
    """
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int

    return num_float


def save_argparse(
    args: argparse.Namespace,
    filename: str,
    exclude: list = None
):
    r"""
    Saves the argparse namespace to a file.

    Parameters
    ----------
    args : argparse.Namespace
        The argparse namespace to save.
    filename : str
        The filename to save the argparse namespace to.
    exclude : list, optional
        A list of keys to exclude from the argparse namespace. The default is None.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")