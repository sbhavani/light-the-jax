"""Command line interface for light_the_jax."""

import pip._internal.main as pip_main

from ._patch import patch

main = patch(pip_main.main)
