"""Command line interface for light_the_jax."""

import pip._internal.main as pip_main
import logging
import sys

from ._patch import patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[light-the-jax] %(message)s",
    stream=sys.stderr,
)

main = patch(pip_main.main)
