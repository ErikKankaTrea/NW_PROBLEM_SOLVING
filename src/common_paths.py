import os
import json

def _norm_path(path):
    """
    Decorator function intended for using it to normalize a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """
    def normalize_path(*args):
        return os.path.normpath(path(*args))
    return normalize_path


def _assure_path_exists(path):
    """
    Decorator function intended for checking the existence of a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """
    def assure_exists(*args):
        assert os.path.exists(path(*args)), "Path {} not found".format(path(*args))
        return path(*args)
    return assure_exists


def _is_output_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an output path retrieval
    function
    """
    @_norm_path
    @_assure_path_exists
    def check_existence_or_create_it(*args):
        if not os.path.exists(path(*args)):
            "Path didn't exist... creating it: {}".format(path(*args))
            os.makedirs(path(*args))
        return path(*args)
    return check_existence_or_create_it


def _is_input_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an input path retrieval
    function
    """
    @_norm_path
    @_assure_path_exists
    def check_existence(*args):
        return path(*args)
    return check_existence


@_is_input_path
def get_data_path():
    with open("./settings.json") as f:
        settings = json.load(f)
    return settings["data_path"]


@_is_output_path
def get_output_path():
    with open("./settings.json") as f:
        settings = json.load(f)
    return settings["output_path"]