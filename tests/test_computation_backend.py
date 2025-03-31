import platform
import subprocess
from unittest import mock

import pip._vendor.packaging.version
import pytest

from light_the_jax._cb import (
    CPUBackend,
    ComputationBackend,
    CUDABackend,
    ROCmBackend,
    _detect_compatible_cuda_backends,
    _detect_nvidia_driver_version,
    detect_compatible_computation_backends,
)

from types import SimpleNamespace

try:
    subprocess.check_call(
        "nvidia-smi",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    NVIDIA_DRIVER_AVAILABLE = True
except subprocess.CalledProcessError:
    NVIDIA_DRIVER_AVAILABLE = False


skip_if_nvidia_driver_unavailable = pytest.mark.skipif(
    not NVIDIA_DRIVER_AVAILABLE, reason="Requires nVidia driver."
)


class GenericComputationBackend(ComputationBackend):
    @property
    def local_specifier(self):
        return "generic"

    def __lt__(self, other):
        return NotImplemented


@pytest.fixture
def generic_backend():
    return GenericComputationBackend()


class TestComputationBackend:
    def test_eq(self, generic_backend):
        assert generic_backend == generic_backend
        assert generic_backend == generic_backend.local_specifier
        assert generic_backend != 0

    def test_hash_smoke(self, generic_backend):
        assert isinstance(hash(generic_backend), int)

    def test_repr_smoke(self, generic_backend):
        assert isinstance(repr(generic_backend), str)

    def test_from_str_cpu(self):
        string = "cpu"
        backend = ComputationBackend.from_str(string)
        assert isinstance(backend, CPUBackend)

    @pytest.mark.parametrize(
        ("major", "minor", "string"),
        [
            pytest.param(major, minor, string, id=string)
            for major, minor, string in (
                (12, 3, "cu123"),
                (12, 3, "cu12.3"),
                (12, 3, "cuda123"),
                (12, 3, "cuda12.3"),
            )
        ],
    )
    def test_from_str_cuda(self, major, minor, string):
        backend = ComputationBackend.from_str(string)
        assert isinstance(backend, CUDABackend)
        assert backend.major == major
        assert backend.minor == minor

    @pytest.mark.parametrize(
        ("major", "minor", "patch", "string"),
        [
            pytest.param(major, minor, patch, string, id=string)
            for major, minor, patch, string in (
                (4, 5, 2, "rocm4.5.2"),
                (5, 0, None, "rocm5.0"),
            )
        ],
    )
    def test_from_str_rocm(self, major, minor, patch, string):
        backend = ComputationBackend.from_str(string)
        assert isinstance(backend, ROCmBackend)
        assert backend.major == major
        assert backend.minor == minor
        assert backend.patch == patch

    @pytest.mark.parametrize("string", (("unknown", "cudnn")))
    def test_from_str_unknown(self, string):
        with pytest.raises(ValueError, match=string):
            ComputationBackend.from_str(string)


class TestCPUBackend:
    def test_eq(self):
        backend = CPUBackend()
        assert backend == "cpu"


class TestCUDABackend:
    def test_eq(self):
        major = 42
        minor = 21
        backend = CUDABackend(major, minor)
        assert backend == f"cu{major}{minor}"


class TestROCmBackend:
    @pytest.mark.parametrize("patch", [10, None])
    def test_eq_with_patch(self, patch):
        major = 42
        minor = 21
        backend = ROCmBackend(major, minor, patch)
        assert (
            backend == f"rocm{major}.{minor}{f'.{patch}' if patch is not None else ''}"
        )


class TestOrdering:
    def test_cpu(self):
        assert CPUBackend() < CUDABackend(0, 0)

    def test_cuda(self):
        assert CUDABackend(0, 0) > CPUBackend()
        assert CUDABackend(1, 2) < CUDABackend(2, 1)
        assert CUDABackend(2, 1) < CUDABackend(10, 0)

    def test_rocm(self):
        assert ROCmBackend(0, 0, 0) > CPUBackend()

        assert ROCmBackend(1, 2, 3) < ROCmBackend(3, 2, 1)
        assert ROCmBackend(3, 2, 1) < ROCmBackend(10, 9, 8)

        assert ROCmBackend(1, 2) < ROCmBackend(1, 2, 0)
        assert ROCmBackend(1, 2, 0) > ROCmBackend(1, 2)

    def test_cuda_vs_rocm(self):
        cuda_backend = CUDABackend(1, 2)
        rocm_backend = ROCmBackend(1, 2)

        with pytest.raises(TypeError):
            cuda_backend < rocm_backend

        with pytest.raises(TypeError):
            rocm_backend < cuda_backend


@pytest.fixture
def patch_nvidia_driver_version(mocker):
    def factory(version):
        return mocker.patch(
            "_detect_nvidia_driver_version",
            return_value=SimpleNamespace(stdout=f"driver_version\n{version}"),
        )

    return factory


def cuda_backends_params():
    params = []
    for system, minimum_driver_versions in _MINIMUM_DRIVER_VERSIONS.items():
        cuda_versions, driver_versions = zip(*sorted(minimum_driver_versions.items()))
        cuda_backends = tuple(
            CUDABackend(version.major, version.minor) for version in cuda_versions
        )

        # latest driver supports every backend
        params.append(
            pytest.param(
                system,
                str(driver_versions[-1]),
                set(cuda_backends),
                id=f"{system.lower()}-latest",
            )
        )

        # outdated driver supports no backend
        params.append(
            pytest.param(
                system,
                str(driver_versions[0].major - 1),
                {},
                id=f"{system.lower()}-outdated",
            )
        )

    return pytest.mark.parametrize(
        ("system", "nvidia_driver_version", "compatible_cuda_backends"), params
    )


class TestDetectCompatibleComputationBackends:
    def test_no_nvidia_driver(self, mocker):
        mocker.patch(
            "_detect_nvidia_driver_version",
            side_effect=subprocess.CalledProcessError(1, ""),
        )

        assert detect_compatible_computation_backends() == {CPUBackend()}

    @cuda_backends_params()
    def test_cuda_backends(
        self,
        mocker,
        patch_nvidia_driver_version,
        system,
        nvidia_driver_version,
        compatible_cuda_backends,
    ):
        mocker.patch("platform.system", return_value=system)
        patch_nvidia_driver_version(nvidia_driver_version)

        backends = detect_compatible_computation_backends()
        assert backends == {CPUBackend(), *compatible_cuda_backends}

    @skip_if_nvidia_driver_unavailable
    def test_cuda_backend(self):
        backend_types = {
            type(backend) for backend in detect_compatible_computation_backends()
        }
        assert CUDABackend in backend_types
