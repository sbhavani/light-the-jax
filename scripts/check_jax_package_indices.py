import itertools
import json

import requests
import tqdm
from bs4 import BeautifulSoup

from light_the_jax._cb import _MINIMUM_DRIVER_VERSIONS, CPUBackend, CUDABackend
from light_the_jax._patch import (
    Channel,
    get_index_urls,
    JAX_DISTRIBUTIONS,
    THIRD_PARTY_PACKAGES,
)

EXCLUDED_JAX_PACKAGES = {
    # Add any JAX packages that should be excluded from the check
    "jax-nightly",
    "jaxlib-nightly",
    "tensorflow-cpu",
    "tensorflow-gpu",
    "tensorflow-rocm",
}
HANDLED_PACKAGES = JAX_DISTRIBUTIONS | THIRD_PARTY_PACKAGES

COMPUTATION_BACKENDS = {
    CUDABackend(cuda_version.major, cuda_version.minor)
    for minimum_driver_versions in _MINIMUM_DRIVER_VERSIONS.values()
    for cuda_version, minimum_driver_version in minimum_driver_versions.items()
}
COMPUTATION_BACKENDS.add(CPUBackend())

INDEX_URLS = sorted(
    set(
        itertools.chain.from_iterable(
            get_index_urls(COMPUTATION_BACKENDS, channel) for channel in iter(Channel)
        )
    )
)


def main():
    available = set()
    for url in tqdm.tqdm(INDEX_URLS):
        response = requests.get(url)
        if not response.ok:
            continue

        soup = BeautifulSoup(response.text, features="html.parser")

        available.update(tag.string for tag in soup.find_all(name="a"))
    available = available - EXCLUDED_JAX_PACKAGES

    print(
        json.dumps(
            dict(
                missing=sorted(available - HANDLED_PACKAGES),
                extra=sorted(HANDLED_PACKAGES - available),
            )
        )
    )


if __name__ == "__main__":
    main()
