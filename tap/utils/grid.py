from itertools import groupby, product
from typing import Mapping

import collections


def linearize(dictionary: Mapping):
    """
    Linearize a nested dictionary making keys, tuples
    :param dictionary: nested dict
    :return: one level dict
    """
    exps = []
    for key, value in dictionary.items():
        if isinstance(value, collections.abc.Mapping):
            exps.extend(
                ((key, lin_key), lin_value) for lin_key, lin_value in linearize(value)
            )
        elif isinstance(value, list):
            exps.append((key, value))
        elif value is None:
            exps.append((key, [{}]))
        else:
            raise ValueError(
                f"Only dict, lists or None!!! -> {value} is {type(value)} for key {key}"
            )
    return exps


def linearized_to_string(lin_dict):
    def linearize_key(key):
        if type(key) == tuple:
            return f"{key[0]}.{linearize_key(key[1])}"
        return key

    return [(linearize_key(key), value) for key, value in lin_dict]


def extract(elem: tuple):
    """
    Exctract the element of a single element tuple
    :param elem: tuple
    :return: element of the tuple if singleton or the tuple itself
    """
    if len(elem) == 1:
        return elem[0]
    return elem


def delinearize(lin_dict):
    """
    Convert a dictionary where tuples can be keys in na nested dictionary
    :param lin_dict: dicionary where keys can be tuples
    :return:
    """
    # Take keys that are tuples
    filtered = list(filter(lambda x: isinstance(x[0], tuple), lin_dict.items()))
    filtered.sort(key=lambda x: x[0][0])
    # Group it to make one level
    grouped = groupby(filtered, lambda x: x[0][0])
    # Create the new dict and apply recursively
    new_dict = {
        k: delinearize({extract(elem[0][1:]): elem[1] for elem in v})
        for k, v in grouped
    }
    # Remove old items and put new ones
    base_values = {k: v for k, v in lin_dict.items() if (k, v) not in filtered}
    delin_dict = {**base_values, **new_dict}
    return delin_dict


def make_grid(dict_of_list, return_cartesian_elements=False):
    """
    Produce a list of dict for each combination of values in the input dict given by the list of values
    :param return_cartesian_elements: if True return the elements that differs from the base dict
    :param dict_of_list: a dictionary where values can be lists
    :params return_cartesian_elements: return elements multiplied

    :return: a list of dictionaries given by the cartesian product of values in the input dictionary
    """
    # Linearize the dict to make the cartesian product straight forward
    linearized_dict = linearize(dict_of_list)
    # Compute the grid
    keys, values = zip(*linearized_dict)
    if any(map(lambda x: len(x) == 0, values)):
        raise ValueError("There shouldn't be empty lists in grid!!")
    grid_dict = list(dict(zip(keys, values_list)) for values_list in product(*values))
    # Delinearize the list of dicts
    grid = [delinearize(dictionary) for dictionary in grid_dict]
    if return_cartesian_elements:
        ce = list(filter(lambda x: len(x[1]) > 1, linearized_dict))
        return grid, ce
    return grid
