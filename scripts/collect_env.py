#!/usr/bin/env python

import itertools
import platform
import subprocess

import importlib_metadata
import pip
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import InvalidVersion, Version


try:
    import light_the_jax
except ModuleNotFoundError:
    light_the_jax = None

NOT_AVAILABLE = "N/A"

# TODO: somehow merge this with light_the_jax._patch.JAX_DISTRIBUTIONS to avoid
#  duplication
JAX_DISTRIBUTIONS = {
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "equinox",
    "orbax",
    "dm-haiku",
    "numpyro",
    "blackjax",
    "diffrax",
}


def main():
    header("System")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")
    nvidia_driver_version = detect_nvidia_driver_version()
    print(f"NVIDIA driver version: {nvidia_driver_version or NOT_AVAILABLE}")

    header("Environment")
    for name, version in itertools.chain(
        [
            ("pip", pip.__version__),
            (
                "light_the_jax",
                light_the_jax.__version__ if light_the_jax else NOT_AVAILABLE,
            ),
        ],
        detect_jax_or_dependent_packages(),
    ):
        print(f"- `{name}=={version}`")


# TODO: somehow merge this with light_the_jax._cb._detect_nvidia_driver_version to
#  avoid duplication
def detect_nvidia_driver_version():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return Version(result.stdout.splitlines()[-1])
    except (FileNotFoundError, subprocess.CalledProcessError, InvalidVersion):
        return None


def detect_jax_or_dependent_packages():
    packages = {
        (dist.name, dist.version)
        for dist in importlib_metadata.distributions()
        if any(
            name in JAX_DISTRIBUTIONS
            for name in itertools.chain(
                [dist.name],
                (
                    [Requirement(req_str).name for req_str in dist.requires]
                    if dist.requires
                    else []
                ),
            )
        )
    }
    return sorted(
        packages,
        key=lambda package: (package[0] not in JAX_DISTRIBUTIONS, package[0]),
    )


def header(name):
    print()
    print(f"#### {name}")
    print()


if __name__ == "__main__":
    main()
