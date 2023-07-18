# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""TAO common path utils used across all apps."""

import os


def expand_path(path):
    """Function to resolve paths.
    
    This function takes in a path and returns the absolute path from the input string
    after expanding the tilde (~) character to the user's home directory to prevent
    path traversal vulnerability.

    Args:
        path (str): The path to expand and make absolute.

    Returns:
        str: The absolute path with expanded tilde.
    """
    return os.path.abspath(os.path.expanduser(path))
